import pandas as pd
import torch
import numpy as np
import os
import os.path as op
import json
import argparse
from utils.llm2vec_encoder import LLM2Vec
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
print(token)

# fix random seed
np.random.seed(0)
torch.manual_seed(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


from huggingface_hub import login
# login(token=token)


dataset_name_mappings = {
    # 5-core filtered datasets
    "Games_5core": "Video_Games/5-core/downstream",
    "Movies_5core": "Movies_and_TV/5-core/downstream",
    "Arts_5core": "Arts_Crafts_and_Sewing/5-core/downstream",
    "Sports_5core": "Sports_and_Outdoors/5-core/downstream",
    "Baby_5core": "Baby_Products/5-core/downstream",
    "Goodreads": "Goodreads/clean",
}


class llm2vec_encoder():
    def __init__(self, model_path, peft_model_name_or_path, bidirectional=False):
        self.model_path = model_path

        if bidirectional:
            self.model = LLM2Vec.from_pretrained(
                model_path,
                peft_model_name_or_path=peft_model_name_or_path,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
                use_auth_token=token,
            )
        else:
            self.model = LLM2Vec.from_pretrained(
                model_path,
                peft_model_name_or_path=peft_model_name_or_path,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
                use_auth_token=token,

                enable_bidirectional=False,
                pooling_mode="eos_token",
            )

    def encode(self, sentences, batch_size):
        return np.asarray(self.model.encode(sentences, batch_size=batch_size))
    
    def encode_with_prompt(self, sentences, batch_size, prompts):
        return np.asarray(self.model.encode(sentences, batch_size=batch_size))


def extract_item_embedding_with_prompts(dataset_name, model_path, peft_path, batch_size, prompt_type, bidirectional=False, save_info=None):
    # Load data here
    raw_dataset_name = dataset_name_mappings[dataset_name]
    with open(f"./data/{raw_dataset_name}/item_titles.json", 'r', encoding='utf-8') as file:
        item_metadata = json.load(file)

    if dataset_name in dataset_name_mappings:
        item_ids = [int(int_id) for int_id in item_metadata.keys()]
        max_item_id = max(item_ids)
        assert 0 not in item_ids, "Item IDs should not contain 0"

        # Add a null item as placeholder for item 0
        item_titles = ["Null"]
        for i in range(1, max_item_id + 1):
            item_titles.append(item_metadata[str(i)])

    else:
        raise ValueError("Invalid dataset name")

    # item_titles = item_titles[:100]
    # print(token)

    model = llm2vec_encoder(model_path, peft_model_name_or_path=peft_path, bidirectional=bool(bidirectional))

    item_infos = np.array(item_titles)
    if prompt_type == "direct":
        prompts = generate_direct_item_prompt_pog(item_infos)
    elif prompt_type == "title":
        prompts = generate_title_item_prompt_pog(item_infos)

    item_llama_embeds = model.encode(prompts, batch_size)

    save_path = f"./item_info/{dataset_name}/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if save_info is not None:
        model_name = f"{save_info}"
    else:
        model_name = model_path.replace("/", "_")
    np.save(op.join(save_path, f"{model_name}_{prompt_type}_item_embs.npy"), item_llama_embeds)


def generate_direct_item_prompt_pog(item_info):
    instruct = "To recommend this item to users, this item can be described as: "
    instructs = np.repeat(instruct, len(item_info))
    prompts = item_info

    outputs = np.concatenate((instructs[:, np.newaxis], prompts[:, np.newaxis]), axis=1)
    return outputs


def generate_title_item_prompt_pog(item_info):
    prompts = item_info
    return prompts


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract item embeddings with prompts.")
    parser.add_argument('--dataset', type=str, default="Arts_5core", help="Name of the dataset")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for processing")
    parser.add_argument('--model_path', type=str, default='./', help="Path to the model")
    parser.add_argument('--peft_path', type=str, default=None, help="Path to the PEFT model")
    parser.add_argument('--item_prompt_type', type=str, default="title", help="Type of item prompt")
    parser.add_argument('--bidirectional', type=int, default=1, help="Bidirectional model")
    parser.add_argument('--save_info', type=str, default="Test-only", help="Save information identifier")
    args = parser.parse_args()

    args = parser.parse_args()
    extract_item_embedding_with_prompts(
        args.dataset,
        args.model_path,
        args.peft_path,
        args.batch_size,
        args.item_prompt_type,
        args.bidirectional,
        args.save_info
    )
