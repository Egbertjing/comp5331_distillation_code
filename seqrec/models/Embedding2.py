import torch 
from torch import nn
from dataclasses import dataclass

@dataclass
class Weight:
    data: None

class Embedding2(nn.Module):
    def __init__(self, adapter, embedding):
        super().__init__()
        self.embedding = embedding
        self.adapter = adapter
        
    def forward(self, indices):
        return self.adapter(self.embedding(indices))

    @property
    def weight(self):
        return Weight(self.adapter(self.embedding.weight.data))
        # return Weight(10)
        
if __name__ == "__main__":
    print(Embedding2(None, None).weight.data)