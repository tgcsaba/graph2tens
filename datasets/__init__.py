from .dataset import Dataset
from .tud import TUDataset
from .citation import Citation
from .utils import GraphFeatureScaler

def load_dataset(dataset_name, **kwargs):
    if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Citation(dataset_name, **kwargs)
    else:
        dataset = TUDataset(dataset_name, **kwargs)
    return dataset