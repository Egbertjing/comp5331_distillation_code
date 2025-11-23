import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
import random
from tqdm import tqdm
import os
import copy
import pickle

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


class PurePromptDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        if not test:
            if sample > 0:
                self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.K = K
        self.dedup = dedup
        self.instructs = [
            f"",
        ]
        self.get_inputs()  
    def __len__(self):
        return len(self.data)


    def generate_example_prompt(self, data_point):
        return f"""{data_point["input"]}"""
    
    def generate_prompt(self, data_point):
        return data_point["input"]


    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        for i in range(L):
            if i == 0:
                history += row['history_item_title'][i]
            else:
                history += ", " + row['history_item_title'][i]
        target_item = str(row['item_title'])
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"{history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        
        prompt = self.generate_prompt(history)
        tokens = self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""
        
        attention_mask = [1] * len(tokens)
        
        
        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "text": prompt,
                # "select_index": select_index,
            }    
        
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        if len(tokens) >= self.max_len:
            print(len(tokens))
        
        
        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }
    

    
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            # print(inputs[-1])
            
        self.inputs = inputs
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]



class DPODataset(Dataset):
    def __init__(self, train_file, info_file, tokenizer, neg_num=3, max_len=2048, sample=-1, test = False, seed=0, category="", dedup=False, negative_sample="cf", dpo=True, hard_negative_file=None):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        if not test:
            if sample > 0:
                self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        with open(info_file, 'r') as f:
            info = f.readlines()
            info = ["\"" + _.split('\t')[0].strip(' ') + "\"\n" for _ in info]
            self.item_name = info
            
        with open(hard_negative_file, 'rb') as f:
            hard_negative_dict = pickle.load(f)
        self.hard_negative_dict = hard_negative_dict
        
        self.neg_num = neg_num
        self.max_len = max_len
        self.category = category
        self.neg_num = neg_num
        self.negative_sample = negative_sample
        self.hard_negative_file = hard_negative_file
        self.dpo = dpo
        # self.K = K
        self.dedup = dedup
        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()  
    def __len__(self):
        return len(self.data)


    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response: 
{data_point["output"]}
"""
    
    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response: 
{data_point["output"]}"""

    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ", \"" + row['history_item_title'][i] + "\""      
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\""
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[0]}
"""
        # tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        
        history = self.get_history(self.data.iloc[idx]) # 从交互数据中拆分出历史交互数据和目标物品
        target_item = history['output']
        history['output'] = ''
        # negative_prompt_ids = copy.deepcopy(tokens)
        # negative_items = [item for item in self.item_name if item != target_item]
        # neg_sam = random.sample(negative_items, self.neg_num)
        if not self.test:
            negative_samples = []
            if self.negative_sample == "cf":
                if idx in self.hard_negative_dict:
                    hard_negative_data = self.hard_negative_dict[idx]
                    item = hard_negative_data['negative_item']
                    weight = hard_negative_data['predict_score']
                    non_zero_idx = np.nonzero(weight)[0]
                    
                    cf_weight = hard_negative_data['cf_score']
                    cf_weight = np.array(cf_weight)
                    # get the ranking index of weight, select the items with the least cf_score as negative samples among non_zero_idx
                    ranking_index = np.argsort(cf_weight[non_zero_idx])
                    cf_num = len(non_zero_idx)
                    # negative_samples = [item[non_zero_idx[ranking_index[i]]] for i in range(cf_num)]
                    if len(non_zero_idx) < self.neg_num:
                        # additional_idx is the index of the items with the least cf_score among zero_idx
                        zero_idx = np.setdiff1d(np.arange(len(weight)), non_zero_idx)
                        ranking_zero_idx = np.argsort(cf_weight[zero_idx])
                        additional_idx = zero_idx[ranking_zero_idx[:self.neg_num - cf_num]]

                        # additional_idx = np.random.choice(np.setdiff1d(np.arange(len(weight)), non_zero_idx), self.neg_num - len(non_zero_idx), replace=False)
                        negative_samples = [item[non_zero_idx[ranking_index[i]]] for i in range(cf_num)] + [item[additional_idx[i]] for i in range(self.neg_num - cf_num)]
                        # print(negative_samples)
                    else:
                        negative_samples = [item[non_zero_idx[ranking_index[i]]] for i in range(self.neg_num)]


                    # print(negative_samples)
                else:
                    return []
                


                
            elif self.negative_sample == "hard":
                if idx in self.hard_negative_dict:
                    hard_negative_data = self.hard_negative_dict[idx]
                    weight = hard_negative_data['predict_score']
                    # find the index of non-zero weight
                    non_zero_idx = np.nonzero(weight)[0]

                    if self.neg_num >= 1:
                    
                        if len(non_zero_idx) < self.neg_num:
                            # randomly sample idx in weight instead of non_zero_idx and add to non_zero_idx
                            additional_idx = np.random.choice(np.setdiff1d(np.arange(len(weight)), non_zero_idx), self.neg_num - len(non_zero_idx), replace=False)
                            non_zero_idx = np.concatenate([non_zero_idx, additional_idx])
                            
                        # randomly sample neg_num non-zero weight
                        num = np.random.choice(non_zero_idx, self.neg_num, replace=False)
                        negative_samples = [str(hard_negative_data['negative_item'][num[i]]) for i in range(self.neg_num)]
                    else:
                        negative_samples = [str(hard_negative_data['negative_item'][non_zero_idx[i]]) for i in range(len(non_zero_idx))]
                else:
                    return []
                
            elif self.negative_sample == "random":
                if idx in self.hard_negative_dict:
                    negative_items = [item for item in self.item_name if item != target_item]
                    negative_samples = random.sample(negative_items, self.neg_num)
                else:
                    return []
            else:
                print("negative_sample is not valid")
                return []

        else:
            if idx not in self.hard_negative_dict:
                return []
            
        if not self.test and negative_samples == []:
            print("negative_samples is empty")
        dic_list = []
        prompt = self.generate_prompt(history)
        if not self.dpo:
            dic["prompt"] = instruction + prompt
            dic["chosen"] = target_item
            for i in range(self.neg_num):
                dic[f"rejected{i}"] = negative_samples[i]
                # dic[f"weight{i}"] = 1 / self.neg_num
                dic[f"weight{i}"] = 1
            dic_list.append(dic)
        else:
            if self.test:
                # for reject in self.hard_negative_dict[idx]['negative_item']:
                #     dic = {
                #         "prompt": instruction + prompt,
                #         "chosen": target_item,
                #         "rejected1": reject,
                #         "weight1": 1
                #     }
                #     dic_list.append(dic)
                dic = {
                    "prompt": instruction + prompt,
                    "chosen": target_item,
                }
                for i, reject in enumerate(self.hard_negative_dict[idx]['negative_item']):
                    dic[f"reject{i+1}"] = reject
                    dic[f"weight{i+1}"] = 1
                    # break
                dic_list.append(dic)
            else:
                for reject in negative_samples:
                    dic = {
                        "prompt": instruction + prompt,
                        "chosen": target_item,
                        "rejected1": reject,
                        "weight1": 1
                    }
                    dic_list.append(dic)
        return dic_list
        # tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        # history["input"] = ""
        
        # attention_mask = [1] * len(tokens)
        
        
        # if self.test:
        #     return {
        #         "input_ids": tokens,
        #         "attention_mask": attention_mask,
                
        #         # "select_index": select_index,
        #     }    
        
        # golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        # input_prompt_len = len(tokens)
        # tokens = tokens + golden_tokens
        # attention_mask = [1] * len(tokens)
        # labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        
        # if len(tokens) >= self.max_len:
        #     print(len(tokens))
        
        
        # return {
        #     "input_ids": tokens[-self.max_len:],
        #     "attention_mask": attention_mask[-self.max_len:],
        #     "labels": labels[-self.max_len:],
            
        # }
    

    
    
    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.pre(i))
            # print(inputs[-1])
            
        self.inputs = inputs
    
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.pre(idx)  