from recdata.RecItemData import RecItemData
from recdata.SeqRecData import SeqRecData
from recdata.ItemTitleData import ItemTitleData


def load_dataset(dataset_name, split="validation", file_path=None, **kwargs):
    """
    Loads a dataset by name.

    Args:
        dataset_name (str): Name of the dataset to load.
        split (str): Split of the dataset to load.
        file_path (str): Path to the dataset file.
    """
    dataset_mapping = {
        "ItemRec": RecItemData,
        "SeqRec": SeqRecData,
        "ItemTitles": ItemTitleData,
    }

    if dataset_name.split("_")[0] not in dataset_mapping:
        raise NotImplementedError(f"Dataset name {dataset_name} not supported.")

    if split not in ["train", "validation", "test"]:
        raise NotImplementedError(f"Split {split} not supported.")

    if "_SeqAug" in dataset_name:
        dataset_name = dataset_name.replace("_SeqAug", "")
        return dataset_mapping[dataset_name](split=split, file_path=file_path, data_augmentation=True, **kwargs)
    else:
        return dataset_mapping[dataset_name](split=split, file_path=file_path, **kwargs)
