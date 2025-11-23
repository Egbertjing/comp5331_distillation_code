import pandas as pd
import torch
import time
import numpy as np
import os
import os.path as op
import json 
from tqdm import tqdm
from baselines.model import EasyRec, Blair, BGE, BERT, GTE_7B, RoBERTa_large_sentence
# Load model directly
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
print(token)

np.random.seed(0)
torch.manual_seed(0)

from huggingface_hub import login
login(token=token)


dataset_name_mappings = {
    # 5-core filtered datasets
    "Games_5core": "Video_Games/5-core/downstream",
    "Movies_5core": "Movies_and_TV/5-core/downstream",
    "Arts_5core": "Arts_Crafts_and_Sewing/5-core/downstream",

    "Sports_5core": "Sports_and_Outdoors/5-core/downstream",
    "Baby_5core": "Baby_Products/5-core/downstream",
    "Goodreads": 'Goodreads/clean',
}


def extract_item_embeddings(model, dataset_name, batch_size, prompt_type, save_info=None):
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

    item_infos = np.array(item_titles)
    if prompt_type == "direct":
        prompts = generate_direct_item_prompt_pog(item_infos)
    elif prompt_type == "title":
        prompts = generate_title_item_prompt_pog(item_infos)    
    # elif prompt_type == "summarize":
    #     prompts = generate_summarize_item_prompt(item_infos)

    item_llama_embeds = []
    
    if type(model).__name__ != "LLM2VecOri":
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
            x = prompts[i:i+batch_size]
            embeds = model(x)
            item_llama_embeds.append(embeds)
        item_llama_embeds = torch.cat(item_llama_embeds, dim=0).cpu().numpy()
    else:
        item_llama_embeds = model(prompts, batch_size)
        

    save_path = f"./item_info/{dataset_name}/"
    os.makedirs(save_path, exist_ok=True)

    if save_info is not None:
        model_name = f"{save_info}"
    else:
        model_name = type(model).__name__
    np.save(op.join(save_path, f"{model_name}_{prompt_type}_item_embs.npy"), item_llama_embeds)


def extract_sequence_embeddings(model, dataset_name, batch_size, prompt_type, save_info=None, max_seq_length=10):
    # Load data here
    raw_dataset_name = dataset_name_mappings[dataset_name]
    with open(f"./data/{raw_dataset_name}/item_titles.json", 'r', encoding='utf-8') as file:
        item_metadata = json.load(file)

    if dataset_name in dataset_name_mappings:
        item_ids = [int(int_id) for int_id in item_metadata.keys()]
        max_item_id = max(item_ids)
        assert 0 not in item_ids, "Item IDs should not contain 0"

    else:
        raise ValueError("Invalid dataset name")
    

    extract_sequence_types = ['train', 'val', 'test']
    for seq_type in extract_sequence_types:
        # Load sequence data
        file_path = f"./data/{raw_dataset_name}/{seq_type}_data.txt"
        with open(file_path, 'r', encoding='utf-8') as file:
            item_seqs = [list(map(int, line.split()))[-max_seq_length-1: -1] for line in file]
            for seq in item_seqs:
                assert len(seq) != 0 and len(seq) <= max_seq_length

        sequence_titles = []
        for item_seq in item_seqs:
            sequnce_titles = "#item {" + "}, #item {".join([item_metadata[str(item_id)] for item_id in item_seq]) + "}"
            title = f"Given the user with the following historical interactions: Predict the next item that this user would like to interact with {sequnce_titles}"
            sequence_titles.append(title)


        item_infos = np.array(sequence_titles)
        prompts = item_infos

        seq_embeds = []

        if type(model).__name__ != "LLM2VecOri":
            for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
                x = prompts[i:i+batch_size]
                embeds = model(x)
                seq_embeds.append(embeds)
            item_llama_embeds = torch.cat(item_llama_embeds, dim=0).cpu().numpy()
        else:
            seq_embeds = model(prompts, batch_size)

        # for i in range(0, len(prompts), batch_size):
        #     x = prompts[i:i+batch_size]
        #     embeds = model(x)
        #     seq_embeds.append(embeds)

        save_path = f"./item_info/{dataset_name}/"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        if save_info is not None:
            model_name = f"{save_info}"
        else:
            model_name = type(model).__name__
        np.save(op.join(save_path, f"{model_name}_{seq_type}_seq_embs.npy"), seq_embeds)


def generate_direct_item_prompt_pog(item_info):
    instruct = "To recommend this fashion item to users, this item can be described as: "
    instructs = np.repeat(instruct, len(item_info))
    prompts = item_info

    outputs = np.concatenate((instructs[:, np.newaxis], prompts[:, np.newaxis]), axis=1)
    return outputs


def generate_title_item_prompt_pog(item_info):
    # instruct = ""
    # instructs = np.repeat(instruct, len(item_info))
    prompts = item_info
    # outputs = np.concatenate((instructs[:, np.newaxis], prompts[:, np.newaxis]), axis=1)
    return prompts

def main(
    model_name="blair", # blair, llm2vec
    mode="item", # or sequence
    dataset_name = "Yelp",
):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "easyrec":
        model = EasyRec(device)
    elif model_name == "blair":
        model = Blair(device)
    elif model_name == "bge":
        model = BGE(device)
    elif model_name == "bert":
        model = BERT(device)
    elif model_name == "gte_7b":
        model = GTE_7B(device)
    elif model_name == "roberta_large_sentence":
        model = RoBERTa_large_sentence(device)
    
    # model.to("cuda")
        
    if mode == "item":
        extract_item_embeddings(model, dataset_name, 64, "title")
    elif mode == "sequence":
        extract_sequence_embeddings(model, dataset_name, 8, "title")
    else:
        raise ValueError("Invalid mode")

if __name__ == "__main__":

    datasets = ["Goodreads"]
    baselines = ["easyrec"]

    for dataset in datasets:
        for baseline in baselines:
            print(f"Processing {baseline} on {dataset}")
            main(model_name=baseline, mode="item", dataset_name=dataset)
    
