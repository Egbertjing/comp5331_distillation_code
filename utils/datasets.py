#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp 
import json
import torch
from torch.utils.data import Dataset, DataLoader


def print_statistics(X, string):
    print('>'*10 + string + '>'*10 )
    print('Average interactions', X.sum(1).mean(0).item())
    nonzero_row_indice, nonzero_col_indice = X.nonzero()
    unique_nonzero_row_indice = np.unique(nonzero_row_indice)
    unique_nonzero_col_indice = np.unique(nonzero_col_indice)
    print('Non-zero rows', len(unique_nonzero_row_indice)/X.shape[0])
    print('Non-zero columns', len(unique_nonzero_col_indice)/X.shape[1])
    print('Matrix density', len(nonzero_row_indice)/(X.shape[0]*X.shape[1]))


class TrainDataset(Dataset):
    def __init__(self, ui_pairs, ui_graph, num_items):
        self.ui_pairs = ui_pairs
        self.ui_graph = ui_graph
        self.num_items = num_items


    def __getitem__(self, index):
        uid, pos_id = self.ui_pairs[index]
        neg_id = np.random.randint(self.num_items)
        while self.ui_graph[uid, neg_id] == 1:
            neg_id = np.random.randint(self.num_items)                                                                                                      
        return uid, pos_id, int(neg_id)


    def __len__(self):
        return len(self.ui_pairs)


class TestDataset(Dataset):
    def __init__(self, ui_pairs, ui_graph, ui_graph_train, num_users, num_items):
        self.ui_pairs = ui_pairs
        self.ui_graph = ui_graph
        self.train_mask_ui = ui_graph_train

        self.num_users = num_users
        self.num_items = num_items


    def __getitem__(self, index):
        ui_grd = torch.from_numpy(self.ui_graph[index].toarray()).squeeze()
        ui_mask = torch.from_numpy(self.train_mask_ui[index].toarray()).squeeze()
        return index, ui_grd, ui_mask


    def __len__(self):
        return self.ui_graph.shape[0]


class Datasets():
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size']
        batch_size_test = conf['test_batch_size']
        self.num_users, self.num_items = self.get_dataset_size()

        ui_pairs_train, ui_graph_train = self.get_graph("train.txt")
        ui_pairs_val, ui_graph_val = self.get_graph("valid.txt")
        ui_pairs_test, ui_graph_test = self.get_graph("test.txt")

        self.ui_graph_train = ui_graph_train

        self.train_data = TrainDataset(ui_pairs_train, ui_graph_train, self.num_items)
        self.val_data = TestDataset(ui_pairs_val, ui_graph_val, ui_graph_train, self.num_users, self.num_items)
        self.test_data = TestDataset(ui_pairs_test, ui_graph_test, ui_graph_train, self.num_users, self.num_items)

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size_train, shuffle=True, num_workers=20, drop_last=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size_test, shuffle=False, num_workers=10)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size_test, shuffle=False, num_workers=10)


    def get_dataset_size(self):
        data_path = self.path
        with open(data_path + "count.json", 'r') as file:
            count_info = json.load(file)

        n_users = count_info['#U']
        n_items = count_info['#I']
        return int(n_users), int(n_items)


    def get_graph(self, filename):
        data_path = self.path

        ui_pairs = []
        with open(data_path + filename, 'r') as file:
            for line in file:
                parts = line.strip().split(" ")
                user_id = parts[0]
                item_ids = parts[1:]
                for item_id in item_ids:
                    ui_pairs.append([int(user_id), int(item_id)])
        ui_pairs = np.array(ui_pairs, dtype=np.int32)

        indice = np.array(ui_pairs, dtype=np.int32)
        values = np.ones(len(ui_pairs), dtype=np.float32)
        ui_graph = sp.coo_matrix((values, (indice[:, 0], indice[:, 1])), shape=(self.num_users, self.num_items)).tocsr()
        return ui_pairs, ui_graph