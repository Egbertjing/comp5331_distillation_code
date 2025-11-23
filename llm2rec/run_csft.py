import os
os.environ["NCCL_P2P_DISABLE"] = "1"  # 禁用 NVLink
os.environ["NCCL_IB_DISABLE"] = "1"   # 禁用 InfiniBand，如果适用
os.environ["NCCL_NET_GDR_LEVEL"] = "0"  # 禁用 GDR（GPU 直连）
os.environ['HF_TOKEN'] = '/data/hjingaa/.cache'
os.environ['HF_HOME'] = '/data/hjingaa/.cache'
import sys
from typing import List
import numpy as np 
import fire

import torch
import transformers
from transformers import EarlyStoppingCallback, AutoConfig
from peft import LoraConfig, get_peft_model
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import PurePromptDataset
def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



def train(
    # model/data params
    base_model: str = "/home/yzhe/workspace/huggingface_data/hub/Qwen2-0.5B",  # the only required argument
    train_file: str="./data/AmazonMix6/5-core/train/AmazonMix-6.csv",
    eval_file: str="./data/AmazonMix6/5-core/valid/AmazonMix-6.csv",
    output_dir: str = "./output/Test-SFT",
    sample: int = -1,
    seed: int = 0,
    
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss./output/Mix2-SFT-${category}
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    use_lora: bool = False,
    
    local_rank: int = 0,
    deepspeed: str ="./deepspeed.json",
    category: str="AmazonMix-6",
    K: int = 0,
    version: str = "base",
    train_from_scratch: bool = False,
):
    os.environ['WANDB_PROJECT'] = wandb_project
    # print(train_file)
    category_dict = {"AmazonMix-6": "items", "Office_Products": "office products", "Books": "books", "Goodreads": "books", "Steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "Industrial_and_Scientific": "industrial and scientific", "Automotive": "automotive products", "Grocery_and_Gourmet_Food": "grocery and gourmet food", "Software": "software", "Pet_Supplies": "pet supply products"}
    print(category)
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # uses.environ["WANDB_LOG_MODEL"] = wandb_log_model
    # os.environ["WANDB_DISABLED"] = "true"
    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            # device_map=device_map,
            trust_remote_code=True,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")
    
    if use_lora:
        # if base model is a Qwen model, use the Qwen model's config
        if "Qwen" in base_model:
            print("Using Qwen model")
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],  # Lora settings for Qwen model
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
        elif "Llama" in base_model:
            print("Using Llama model")
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],  # Lora settings for Llama model
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

        # Wrap the model with LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    train_data = PurePromptDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category, K = K)
    # val_data = PurePromptDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, category=category, K = K)
    val_data = PurePromptDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=2000, category=category, K = K)
        
    print("LOAD DATA FINISHED")    
    
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    from datasets import Dataset as HFDataset
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_train_dataset = hf_train_dataset.shuffle(seed=seed)

    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})
    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=200,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            # evaluation_strategy="epoch",
            # save_strategy="epoch",
            max_steps=10000,
            evaluation_strategy="steps",       # Changed from "epoch" to "steps"
            eval_steps=2000,                   # Evaluate every 1000 steps
            save_strategy="steps",             # Changed from "epoch" to "steps"
            save_steps=2000,                   # Save checkpoint every 1000 steps

            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb",
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
        # optimizers=(optimizer, lr_scheduler) 
    )
    model.config.use_cache = False
    trainer.evaluate()
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if use_lora:
        model.save_pretrained(output_dir, safe_serialization=True)
    else:
        model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)