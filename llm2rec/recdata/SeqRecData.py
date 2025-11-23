import json
import random
import os
import pandas as pd

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

AMAZON_TRAIN_DATA_PATH_MAPPING = {
    "Arts": "Arts_Crafts_and_Sewing/5-core/train/Arts_Crafts_and_Sewing_5_2014-9-2023-10.csv",
    "Electronics": "Electronics/5-core/train/Electronics_5_2016-9-2023-10.csv",
    "Home": "Home_and_Kitchen/5-core/train/Home_and_Kitchen_5_2016-9-2023-10.csv",
    "Movies": "Movies_and_TV/5-core/train/Movies_and_TV_5_2019-9-2023-10.csv",
    "Tools": "Tools_and_Home_Improvement/5-core/train/Tools_and_Home_Improvement_5_2016-9-2023-10.csv",
    "Games": "Video_Games/5-core/train/Video_Games_5_1996-9-2023-10.csv",
}

AMAZON_ITEM_INFO_MAPPING = {
    "Arts": "Arts_Crafts_and_Sewing/5-core/downstream/item_titles.json",
    "Electronics": "Electronics/5-core/downstream/item_titles.json",
    "Home": "Home_and_Kitchen/5-core/downstream/item_titles.json",
    "Movies": "Movies_and_TV/5-core/downstream/item_titles.json",
    "Tools": "Tools_and_Home_Improvement/5-core/downstream/item_titles.json",
    "Games": "Video_Games/5-core/downstream/item_titles.json",
}

NUM_TRAINING_SAMPLES = 100000


class SeqRecData(Dataset):
    def __init__(
        self,
        dataset_name: str = "Rec",
        split: str = "validation",
        file_path: str = "./data",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        data_augmentation: bool = False,
        augmentation_rate: float = 0.2,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.data_augmentation = data_augmentation
        self.augmentation_rate = augmentation_rate

        # list storing all item titles for random negative sampling
        self.negative_item_pool = []

        self.data = []
        self.load_data(file_path)

        # remove NoneType samples
        self.negative_item_pool = [item for item in self.negative_item_pool if item is not None]


    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        logger.info(f"Loading SeqRec data from {file_path}...")
        # file path is actually a directory

        data_map = {}
        all_samples = []
        id_ = 0
        for dataset in AMAZON_TRAIN_DATA_PATH_MAPPING:
            logger.info(f"Loading dataset {dataset}...")
            if dataset not in data_map:
                data_map[dataset] = []
            
            dataset_samples = self.process_data(file_path, dataset, self.data_augmentation, self.augmentation_rate)

            for i, sample in enumerate(dataset_samples):
                query = self.separator + sample['query']
                pos = self.separator + sample["positive"]
                neg = self.separator + sample["negative"]

                data_map[dataset].append(id_)
                self.negative_item_pool.append(neg)

                if not self.data_augmentation:
                    all_samples.append(
                        DataSample(
                            id_=id_,
                            query=query,
                            positive=pos,
                            negative=neg,
                            task_name=dataset,
                        )
                    )
                else:
                    aug_query = self.separator + sample["aug_query"]
                    all_samples.append(
                        DataSample(
                            id_=id_,
                            query=query,
                            positive=pos,
                            negative=neg,
                            task_name=dataset,
                            aug_query=aug_query,
                        )
                    )
                id_ += 1

        # combine split1 and split2
        new_data_map = {}
        for dataset in data_map:
            new_dataset = dataset.replace("_split1", "").replace("_split2", "")
            if new_dataset not in new_data_map:
                new_data_map[new_dataset] = []
            new_data_map[new_dataset] += data_map[dataset]
        data_map = new_data_map

        if self.shuffle_individual_datasets:
            for task, samples in data_map.items():
                random.shuffle(samples)

        datasets = list(data_map.keys())

        logger.info(
            f"Batching REC data properly for effective batch size of {self.effective_batch_size}..."
        )
        all_batches = []
        for dataset in datasets:
            dataset_samples = data_map[dataset]
            for i in range(0, len(dataset_samples), self.effective_batch_size):
                batch = dataset_samples[i : i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    all_batches.append(batch)
                else:
                    logger.info(f"Skip 1 batch for dataset {dataset}.")
        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)

        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            if not self.data_augmentation:
                return TrainSample(
                    texts=[sample.query, sample.positive, sample.negative], label=1.0
                )
            else:
                return TrainSample(
                    texts=[sample.query, sample.positive, sample.negative, sample.aug_query], label=1.0
                )
    
        elif self.split == "validation":
            assert False, "SeqRecData does not have a validation split."


    def process_data(self, file_path, dataset_name, data_augmentation=False, augmentation_rate=0.2):
        item_info_path = AMAZON_ITEM_INFO_MAPPING[dataset_name]
        item_info_path = os.path.join(file_path, item_info_path)
        with open(item_info_path, "r") as f:
            item_info = json.load(f)
        # change key of item_info from string to int
        assert "0" not in item_info

        # The item_info is a dictionary with keys starting from 1. We need to change the keys to start from 0
        item_info = {int(k) - 1: v for k, v in item_info.items()}
        candidiate_item_ids = list(item_info.keys())

        train_data_path = AMAZON_TRAIN_DATA_PATH_MAPPING[dataset_name]
        train_data_path = os.path.join(file_path, train_data_path)
        dataset_samples = pd.read_csv(train_data_path)

        # random sample a fixed number of samples
        dataset_samples = dataset_samples.sample(n=NUM_TRAINING_SAMPLES, random_state=42)

        # interate through all samples
        samples = []
        for i, row in dataset_samples.iterrows():
            his_ids = eval(row["history_item_id"])
            pos_id = row["item_id"]
            neg_id = random.choice(candidiate_item_ids)
            while neg_id == pos_id or neg_id in his_ids:
                neg_id = random.choice(candidiate_item_ids)

            his_titles = eval(row["history_item_title"])
            pos_title = row["item_title"]
            neg_title = item_info[neg_id]

            if data_augmentation:
                if len(his_ids) <= 2:
                    aug_his_ids = his_ids
                else:
                    num_items_to_drop = int(len(his_ids) * augmentation_rate)
                    if num_items_to_drop == 0:
                        num_items_to_drop = 1
                    
                    remaining_ids = random.sample(his_ids, len(his_ids) - num_items_to_drop)
                    aug_his_ids = [item_id for item_id in his_ids if item_id in remaining_ids]

                aug_his_titles = [item_info[item_id] for item_id in aug_his_ids]


            samples.append({
                "query": ", ".join(his_titles),
                "positive": pos_title,
                "negative": neg_title,
                "aug_query": ", ".join(aug_his_titles) if data_augmentation else None,
            })
        
        return samples


    def generate_negative_samples(self, num_samples):
        return random.sample(self.negative_item_pool, num_samples)
