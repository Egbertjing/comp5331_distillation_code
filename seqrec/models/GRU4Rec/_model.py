import torch
import torch.nn as nn
import numpy as np
from seqrec.base import AbstractModel
from seqrec.modules import in_batch_negative_sampling, extract_axis_1
from ..Embedding2 import Embedding2


class GRU4Rec(AbstractModel):
    def __init__(self, config: dict, pretrained_item_embeddings=None):
        super(GRU4Rec, self).__init__(config=config)
        self.config = config
        self.load_item_embeddings(pretrained_item_embeddings)

        self.gru_layers = nn.GRU(
            input_size=config['hidden_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['layer_num'],
            bias=False,
            batch_first=True,
        )
        self.emb_dropout = nn.Dropout(config['dropout'])

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
        else:
            more_token = 0
            self.pretrained_item_embeddings = nn.Embedding.from_pretrained(
                torch.cat([
                    pretrained_embs[:self.config['item_num']+1],
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
            
            self.item_embedding_pretrained = True
            
            self.item_embeddings = Embedding2(self.item_embeddings_adapter, self.pretrained_item_embeddings)

        if self.config.get('aug', None) == 'sub':
            self.category_embedding = nn.Embedding(self.config['sub_head'],
                                               self.config['hidden_size'] // self.config['sub_head'])
            nn.init.normal_(self.category_embedding.weight, 0, 1)
        

    def get_embeddings(self, items):
        if self.config.get('aug', None) == 'sub':
            return self.item_embeddings(items) * self.get_sub_embeddings(items)
        else:
            return self.item_embeddings(items)

    def get_all_embeddings(self, device=None):
        if self.config.get('aug', None) == 'sub':
            return self.item_embeddings.weight * self.get_sub_embeddings(self.item_embeddings.weight)
        else:
            return self.item_embeddings.weight.data

    def get_current_embeddings(self, device=None):
        if self.config.get('aug', None) == 'sub':
           item_embeddings = self.item_embeddings.weight * self.get_sub_embeddings(self.item_embeddings.weight)
           return item_embeddings[self.config['select_pool'][0]:self.config['select_pool'][1]]
        else:
            return self.item_embeddings.weight.data[self.config['select_pool'][0]:self.config['select_pool'][1]]


    def get_representation(self, batch):
        inputs_emb = self.get_embeddings(batch['item_seqs'])
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(batch['item_seqs'], 0).float().unsqueeze(-1).to(inputs_emb.device)
        seq = seq * mask
        seq, _ = self.gru_layers(seq)
        state_hidden = extract_axis_1(seq, batch['seq_lengths'] - 1).squeeze()
        return state_hidden

    def forward(self, batch):
        state_hidden = self.get_representation(batch)
        test_item_emb = self.get_all_embeddings(state_hidden.device)
        if self.config['loss_type'] == 'bce':
            labels_neg = self._generate_negative_samples(batch)
            labels_neg = labels_neg.view(-1, 1)
            logits = torch.matmul(state_hidden, test_item_emb.transpose(0, 1))
            pos_scores = torch.gather(logits, 1, batch['labels'].view(-1, 1))
            neg_scores = torch.gather(logits, 1, labels_neg)
            pos_labels = torch.ones((batch['labels'].view(-1).shape[0], 1))
            neg_labels = torch.zeros((batch['labels'].view(-1).shape[0], 1))

            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((pos_labels, neg_labels), 0)
            labels = labels.to(state_hidden.device)
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
        if self.config['sample_func'] == 'batch':
            return in_batch_negative_sampling(batch['labels'])

        labels_neg = []
        for index in range(len(batch['labels'])):
            import numpy as np
            neg_samples = np.random.choice(range(self.config['select_pool'][0], self.config['select_pool'][1]), size=1,
                                           replace=False)
            neg_samples = neg_samples[neg_samples != batch['labels'][index]]
            labels_neg.append(neg_samples.tolist())
        return torch.LongTensor(labels_neg).to(batch['labels'].device).reshape(-1, 1)
