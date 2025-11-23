import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Sampler


class SequenceDataset(Dataset):
    def __init__(self, config, sequences, seq_type=None):
        self.sequences = sequences
        self.config = config
        self.seq_type = seq_type

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        item_seq = seq[:-1]
        labels = seq[-1]
        seq_length = len(item_seq)
        padding_length = self.config['max_seq_length'] - len(item_seq)
        if padding_length > 0:
            item_seq = item_seq + [0] * padding_length  # 在后面填充0
        return {
            'item_seqs': torch.tensor(item_seq, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'seq_lengths': seq_length,

            # The variables below are used for sequential embedding generation. Ignore if not needed.
            'seq_ids': idx,
            'seq_type': self.seq_type
        }


class NormalRecData:
    def __init__(self, config: dict):
        self.config = config

    def load_data(self):
        from pathlib import Path

        source_dict = {
            "Goodreads": 'Goodreads/clean',
            "Games_5core": "Video_Games/5-core/downstream",
            "Movies_5core": "Movies_and_TV/5-core/downstream",
            "Arts_5core": "Arts_Crafts_and_Sewing/5-core/downstream",
            "Sports_5core": "Sports_and_Outdoors/5-core/downstream",
            "Baby_5core": "Baby_Products/5-core/downstream",
        }
        self.config['source_dict'] = source_dict

        def read_data_from_file(domain, mode=''):
            base_path = Path('data/')
            file_path = base_path / source_dict[domain] / '{}data.txt'.format(mode)
            with file_path.open('r') as file:
                item_seqs = [list(map(int, line.split()))[-self.config['max_seq_length']-1:] for line in file]

            if mode == '':
                flat_list = [item for sublist in item_seqs for item in sublist]
                import numpy as np
                item_num = np.max(flat_list)
                return item_seqs, item_num
            else:
                return item_seqs

        train_data = []
        valid_data = []
        test_data = []

        tmp_item_seqs, total_item_num = read_data_from_file(self.config['dataset'])
        tmp_train_item_seqs, tmp_valid_item_seqs, tmp_test_item_seqs = (
            read_data_from_file(self.config['dataset'], mode='train_'),
            read_data_from_file(self.config['dataset'], mode='val_'),
            read_data_from_file(self.config['dataset'], mode='test_')
            )
        train_data.extend(tmp_train_item_seqs)
        valid_data.extend(tmp_valid_item_seqs)
        test_data.extend(tmp_test_item_seqs)
        select_pool = [1, total_item_num + 1]

        return (
            SequenceDataset(self.config, train_data, seq_type='train'),
            SequenceDataset(self.config, valid_data, seq_type='val'),
            SequenceDataset(self.config, test_data, seq_type='test'),
            select_pool,
            total_item_num
        )
