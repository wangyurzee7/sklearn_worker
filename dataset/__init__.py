from .JsonLoader import JsonLoader
from .CsvLoader import CsvLoader

dataset_list = {
    "json": JsonLoader,
    "csv": CsvLoader,
}


def get_dataset_loader(dataset_name,conf):
    if dataset_name in dataset_list.keys():
        return dataset_list[dataset_name](conf)
    else:
        raise NotImplementedError