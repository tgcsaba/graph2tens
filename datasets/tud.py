import numpy as np
from spektral import data, datasets, transforms

from sklearn.model_selection import train_test_split

from tensorflow.keras import losses, metrics

from .dataset import Dataset
from .utils import GraphFeatureScaler

# ----------------------------------------------------------------------------------------------------

class TUDataset(Dataset):
    
    def __init__(self, dataset_name, ratio_va=0.1, ratio_te=0.1, batch_size=None, split_id=None):
        self.dataset_name = dataset_name.upper()
        self.ratio_va = ratio_va
        self.ratio_te = ratio_te
        self.batch_size = batch_size
        self.split_id = split_id
        self.dataset = datasets.TUDataset(self.dataset_name)
        self._split = False
        
    @property
    def idx_tr(self):
        if self._split:
            return self._idx_tr
        else:
            return list(range(len(self.dataset)))
    
    @property
    def idx_va(self):
        if self._split:
            return self._idx_va
        else:
            return []
    
    @property
    def idx_te(self):
        if self._split:
            return self._idx_te
        else:
            return []
        
    @property
    def has_edge_feat(self):
        return self.dataset[0].e is not None
        
    def split(self):
        y = np.stack([g.y for g in self.dataset])
        y = np.argmax(y, axis=1)    
        idx_tr, idx_va = train_test_split(
            np.arange(y.shape[0]), test_size=self.ratio_va, stratify=y,
            random_state=self.split_id)
        idx_tr, idx_te = train_test_split(
            idx_tr, test_size=1./(1-self.ratio_va)*self.ratio_te, stratify=y[idx_tr],
            random_state=self.split_id)
        self._idx_tr, self._idx_va, self._idx_te = idx_tr, idx_va, idx_te
        self._split = True 
        return self
    
    def preprocess(self):
        # if no node features add one-hot degree
        if self.dataset[0].x is None:
            max_degree = int(np.max([np.max(g.a.sum(axis=1)) for g in dataset]))
            self.dataset.apply(transforms.Degree(max_degree))
        # normalize node features
        node_scaler = GraphFeatureScaler().fit(self.dataset[self.idx_tr])
        self.dataset = node_scaler.transform(self.dataset)
        if self.has_edge_feat:
            edge_scaler = GraphFeatureScaler(edge_feat=True).fit(self.dataset[self.idx_tr])
            self.dataset = scaler.transform(self.dataset)
        return self
    
    @property
    def loaders(self):
        loader_tr = data.DisjointLoader(self.dataset[self.idx_tr], shuffle=True,
                                   batch_size=self.batch_size or len(self.idx_tr))
        if not self._split:
            return loader_tr
        loader_va = data.DisjointLoader(self.dataset[self.idx_va], shuffle=False,
                                    batch_size=self.batch_size or len(self.idx_va))
        loader_te = data.DisjointLoader(self.dataset[self.idx_te], shuffle=False,
                                    batch_size=self.batch_size or len(self.idx_te))
        return loader_tr, loader_va, loader_te
    
    @property
    def loss(self):
        return losses.CategoricalCrossentropy(reduction='sum')
    
    @property
    def metrics(self):
        return [
        metrics.CategoricalAccuracy(name='acc'),
        metrics.AUC(curve='PR', name='auprc'),
        metrics.AUC(curve='ROC', name='auroc')
    ]

# ----------------------------------------------------------------------------------------------------