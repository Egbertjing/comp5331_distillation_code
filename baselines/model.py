import sys
sys.path.append("baselines/EasyRec")
import torch
from baselines.EasyRecModel import Easyrec_encoder
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer



class EasyRec(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.config = AutoConfig.from_pretrained("hkuds/easyrec-roberta-large")
        self.model = Easyrec_encoder.from_pretrained("hkuds/easyrec-roberta-large", config=self.config,).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("hkuds/easyrec-roberta-large", use_fast=False,)

    def forward(self, x):
        # x is a batch of text sequences
        
        inputs = self.tokenizer(x.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            embeddings = self.model.encode(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        embeddings = F.normalize(embeddings.pooler_output.detach().float(), dim=-1)

        return embeddings
    
class Blair(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("hyp1231/blair-roberta-base")
        self.model = AutoModel.from_pretrained("hyp1231/blair-roberta-base").to(self.device)
        
    def forward(self, x):
        # x is a batch of text sequences
        
        inputs = self.tokenizer(x.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs, return_dict=True).last_hidden_state[:, 0]
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        return embeddings
    
    
class BGE(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5') # or BAAI/bge-m3, BAAI/llm-embedder
        self.model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5').to(self.device)
        self.model.eval()
        
    def forward(self, x):
        # x is a batch of text sequences
        inputs = self.tokenizer(x.tolist(), padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(**inputs)[0][:, 0]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings
    

class BERT(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Use BERT tokenizer
        self.model = AutoModel.from_pretrained('bert-base-uncased').to(self.device)  # Load BERT model
        self.model.eval()  # Set model to evaluation mode

    def forward(self, x):
        # x is a batch of text sequences (list of strings)
        inputs = self.tokenizer(
            x.tolist(), 
            padding=True, 
            truncation=True, 
            return_tensors='pt'  # Return PyTorch tensors
        ).to(self.device)
        
        with torch.no_grad():
            # Pass the tokenized inputs through the BERT model
            outputs = self.model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            normalized_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)

        return normalized_embeddings


class RoBERTa_large_sentence(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')  # Use RoBERTa tokenizer
        self.model = RobertaModel.from_pretrained('roberta-large').to(self.device)  # Load RoBERTa model
        self.model.eval()  # Set model to evaluation mode


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def forward(self, x):
        # x is a batch of text sequences (list of strings)
        inputs = self.tokenizer(
            x.tolist(), 
            padding=True, 
            truncation=True, 
            return_tensors='pt'  # Return PyTorch tensors
        ).to(self.device)
        
        with torch.no_grad():
            # Pass the tokenized inputs through the RoBERTa model
            outputs = self.model(**inputs)
            sentence_embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
            normalized_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return normalized_embeddings
    


class GTE_7B(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct')  # Use GTE tokenizer
        self.model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct').to(self.device)  # Load GTE model
        self.model.eval()  # Set model to evaluation mode


    def last_token_pool(self, last_hidden_states, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), sequence_lengths]


    def forward(self, x):
        # x is a batch of text sequences (list of strings)
        inputs = self.tokenizer(
            x.tolist(), 
            padding=True, 
            truncation=True, 
            return_tensors='pt'  # Return PyTorch tensors
        ).to(self.device)
        
        with torch.no_grad():
            # Pass the tokenized inputs through the GTE model
            outputs = self.model(**inputs)
            pooled_embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            normalized_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        return normalized_embeddings




