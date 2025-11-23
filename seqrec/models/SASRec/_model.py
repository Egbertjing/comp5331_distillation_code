import torch
import torch.nn as nn
import numpy as np
from seqrec.base import AbstractModel
from seqrec.modules import TransformerEncoder_v2, get_attention_mask, gather_indexes
from ..Embedding2 import Embedding2



class SASRec(AbstractModel):
    def __init__(self, config: dict, pretrained_item_embeddings=None):
        super(SASRec, self).__init__(config=config)
        self.config = config
        self.load_item_embeddings(pretrained_item_embeddings)

        # Initialize embeddings
        self.positional_embeddings = nn.Embedding(
            num_embeddings=config['max_seq_length'],
            embedding_dim=config['hidden_size']
        )

        self.emb_dropout = nn.Dropout(config['dropout'])

        # Initialize Transformer layers
        self.transformer_encoder = TransformerEncoder_v2(config)

        # Initialize loss function
        if config['loss_type'] == 'bce':
            self.loss_func = nn.BCEWithLogitsLoss()
        elif config['loss_type'] == "ce":
            self.loss_func = nn.CrossEntropyLoss()

    def load_item_embeddings(self, pretrained_embs):
        if pretrained_embs is None:
            self.item_embeddings = nn.Embedding(
                num_embeddings=self.config['item_num'] + 1,
                embedding_dim=self.config['hidden_size'],
                padding_idx=0
            )
            nn.init.normal_(self.item_embeddings.weight, 0, 1)

        # use pretrained textual embedding with linear mapping as item embedding
        else:
            more_token = 0
            assert pretrained_embs.shape[0] == self.config['item_num'] + 1
            self.pretrained_item_embeddings = nn.Embedding.from_pretrained(
                torch.cat([
                    pretrained_embs,
                    torch.randn(more_token, pretrained_embs.shape[-1]).to(pretrained_embs.device)
                    ]),
                padding_idx=0
            )
            # fix pretrained item embedding
            self.pretrained_item_embeddings.weight.requires_grad = False
            self.pretrained_item_embeddings.weight[-more_token:].requires_grad = True

            assert self.config['adapter_dims'][-1] == -1
            mlp_dims = [self.pretrained_item_embeddings.embedding_dim] + self.config['adapter_dims']
            mlp_dims[-1] = self.config['hidden_size']

            # create mlp with linears and activations
            self.item_embeddings_adapter = nn.Sequential()
            self.item_embeddings_adapter.add_module('linear_0', nn.Linear(mlp_dims[0], mlp_dims[1]))
            for i in range(1, len(mlp_dims) - 1):
                self.item_embeddings_adapter.add_module(f'activation_{i}', nn.ReLU())
                self.item_embeddings_adapter.add_module(f'linear_{i}', nn.Linear(mlp_dims[i], mlp_dims[i + 1]))

            # initialize the adapter
            for name, param in self.item_embeddings_adapter.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
            
            self.item_embeddings = Embedding2(self.item_embeddings_adapter, self.pretrained_item_embeddings)


    # Note: to replace item_embedding, we need to modify both get_embeddings and get_all_embeddings functions.
    def get_embeddings(self, items):
        return self.item_embeddings(items)

    def get_all_embeddings(self, device=None):
        return self.item_embeddings.weight.data


    def get_representation(self, batch):
        inputs_emb = self.get_embeddings(batch['item_seqs'])
        inputs_emb += self.positional_embeddings(
            torch.arange(self.config['max_seq_length']).to(inputs_emb.device)
        )
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(batch['item_seqs'], 0).float().to(inputs_emb.device)
        
        mask = get_attention_mask(mask, bidirectional=False)
        
        seq = self.transformer_encoder(seq, attention_mask=mask)

        output = seq[-1]
        output = gather_indexes(output, batch['seq_lengths'] - 1)
        return output

    def forward(self, batch):
        state_hidden = self.get_representation(batch)
        test_item_emb = self.get_all_embeddings(state_hidden.device)
        if self.config['loss_type'] == 'bce':
            labels_neg = self._generate_negative_samples(batch)
            # labels_neg = labels_neg.view(-1, 1)
            logits = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))
            pos_scores = torch.gather(logits, 1, batch['labels'].view(-1, 1))
            neg_scores = torch.gather(logits, 1, labels_neg)
            pos_labels = torch.ones((batch['labels'].shape[0], 1), device=state_hidden.device)
            neg_labels = torch.zeros((batch['labels'].shape[0], labels_neg.shape[1]), device=state_hidden.device)

            scores = torch.cat((pos_scores, neg_scores), dim=1).view(-1, 1)  # Shape: (batch_size * (1 + num_neg), 1)
            labels = torch.cat((pos_labels, neg_labels), dim=1).view(-1, 1)  # Shape: (batch_size * (1 + num_neg), 1)

            loss = self.loss_func(scores, labels)

        elif self.config['loss_type'] == 'ce':
            logits = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))
            loss = self.loss_func(logits, batch['labels'].view(-1))

        return {'loss': loss}

    def predict(self, batch, n_return_sequences=1):
        state_hidden = self.get_representation(batch).view(-1, self.config['hidden_size'])
        test_item_emb = self.get_all_embeddings(state_hidden.device)
        scores = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))[:,
                 self.config['select_pool'][0]: self.config['select_pool'][1]]
        preds = scores.topk(n_return_sequences, dim=-1).indices + self.config['select_pool'][0]
        return preds

    def _generate_negative_samples(self, batch):
        # if self.config['sample_func'] == 'batch':
        #     return in_batch_negative_sampling(batch['labels'])

        target_neg = []
        for index in range(len(batch['labels'])):
            neg=np.random.randint(self.config['select_pool'][0], self.config['select_pool'][1])
            while neg==batch['labels'][index]:
                neg = np.random.randint(self.config['select_pool'][0], self.config['select_pool'][1])
            target_neg.append(neg)

        return torch.LongTensor(target_neg).to(batch['labels'].device).reshape(-1, 1)
