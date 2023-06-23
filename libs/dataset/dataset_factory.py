from .MQ_dataset import MQDataset
from .online_dataset import OnlineDataset

dataset_factory = {
    'MQ': MQDataset,
    'Online': OnlineDataset
}

def get_dataset(dataset):
    return dataset_factory[dataset]