import numpy as np
from spektral import data, datasets, transforms

from sklearn.model_selection import train_test_split

from tensorflow.keras import losses, metrics

from .dataset import Dataset
from .utils import GraphFeatureScaler, _idx_to_mask, _mask_to_weights

import networkx as nx

# ----------------------------------------------------------------------------------------------------

class Citation(Dataset):
    
    def __init__(self, dataset_name, n_tr_per_class=20, n_va_per_class=30, khop=None, split_id=None):
        
        self.dataset_name = dataset_name.lower()
        self.n_tr_per_class = n_tr_per_class
        self.n_va_per_class = n_va_per_class
        self.khop = khop
        self.split_id = split_id
        self.dataset = datasets.Citation(self.dataset_name)
        self._split = False
        
    @property
    def n_nodes(self):
        return self.dataset[0].y.shape[0]
    
    @property
    def idx_tr(self):
        if self._split:
            return self._idx_tr
        else:
            return list(range(self.n_nodes))
    
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
        y = self.dataset[0].y
        n_classes = y.shape[1]
        y = np.argmax(y, axis=1)
        idx = np.arange(y.shape[0])
        
        if self.khop is not None and self.khop > 0:
            A = self.dataset[0].a.todense()
            np.fill_diagonal(A, 1) 
            # add self-loops so that A**k is nonzero for any lower degree neighbours
            A = A**self.khop > 0 # k-th degree adjacency matrix and remove edge weights
            np.fill_diagonal(A, 0) # remove self-loops
            idx = nx.maximal_independent_set(nx.Graph(A), seed=self.split_id)
        
        idx_tr, idx_te = train_test_split(
            idx, train_size=self.n_tr_per_class * n_classes, stratify=y[idx],
            random_state=self.split_id)
        idx_va, idx_te = train_test_split(
            idx_te, train_size=self.n_va_per_class * n_classes, stratify=y[idx_te],
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
        self.dataset = GraphFeatureScaler().fit_transform(self.dataset)
        if self.has_edge_feat:
            edge_scaler = GraphFeatureScaler(edge_feat=True).fit_transform(self.dataset)
        return self
    
    @property
    def loaders(self):
        loader_tr = data.SingleLoader(
            self.dataset, sample_weights=_mask_to_weights(_idx_to_mask(self.idx_tr, self.n_nodes))
            if self._split else None)
        if not self._split:
            return loader_tr
        loader_va = data.SingleLoader(
            self.dataset, sample_weights=_mask_to_weights(_idx_to_mask(self.idx_va, self.n_nodes)))
        loader_te = data.SingleLoader(
            self.dataset, sample_weights=_mask_to_weights(_idx_to_mask(self.idx_te, self.n_nodes)))
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