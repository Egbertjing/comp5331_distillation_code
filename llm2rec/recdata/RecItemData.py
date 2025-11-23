import json
import random
import os

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


AMAZON_DATASET_NAME_MAPPING = {
    "Arts": "Arts_Crafts_and_Sewing",
    "Electronics": "Electronics",
    "Home": "Home_and_Kitchen",
    "Movies": "Movies_and_TV",
    "Tools": "Tools_and_Home_Improvement",
    "Games": "Video_Games",

    # "Sports": "Sports_and_Outdoors",
}
NUM_TRAINING_SAMPLES = 100000

class RecItemData(Dataset):
    def __init__(
        self,
        dataset_name: str = "Rec",
        split: str = "validation",
        file_path: str = "dataset/llm2vec",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        logger.info(f"Loading Rec data from {file_path}...")
        # file path is actually a directory

        data_map = {}
        all_samples = []
        id_ = 0
        for dataset in AMAZON_DATASET_NAME_MAPPING:
            logger.info(f"Loading dataset {dataset}...")
            if dataset not in data_map:
                data_map[dataset] = []

            dataset_raw_naming = AMAZON_DATASET_NAME_MAPPING[dataset]
            with open(os.path.join(file_path, f"{dataset_raw_naming}/training_item_pairs_gap24.jsonl"), "r") as f:
                dataset_samples = json.loads(f.read().strip())
            
            if len(dataset_samples) > NUM_TRAINING_SAMPLES:
                dataset_samples = random.sample(dataset_samples, NUM_TRAINING_SAMPLES)

            for i, sample in enumerate(dataset_samples):
                query = self.separator + sample[0]
                pos = self.separator + sample[1]
                data_map[dataset].append(id_)

                all_samples.append(
                    DataSample(
                        id_=id_,
                        query=query,
                        positive=pos,
                        task_name=dataset,
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
            return TrainSample(
                texts=[sample.query, sample.positive], label=1.0
            )
        elif self.split == "validation":
            assert False, "RecData does not have a validation split."
