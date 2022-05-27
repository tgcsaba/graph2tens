import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ----------------------------------------------------------------------------------------------------

def _mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)

# ----------------------------------------------------------------------------------------------------

def _idx_to_mask(idx, n):
    mask = np.zeros(n)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

# ----------------------------------------------------------------------------------------------------

class GraphFeatureScaler(TransformerMixin, BaseEstimator):
    """
    Normalizer for continuous-binary mixed graph node (and edge) features.
    """
    def __init__(self, edge_feat=False):
        self.edge_feat = edge_feat
        self._scaler_nonneg = MinMaxScaler() # do not center nonnegative attributes
        self._scaler_sign = StandardScaler()
            
    def _reset(self):
        self._scaler_nonneg._reset()
        self._scaler_sign._reset()
            

    def fit(self, dataset, y=None):
        if self.edge_feat:
            assert dataset[0].e is not None
        self._reset()
            
        feat = np.concatenate([g.e for g in dataset], axis=0) \
                if self.edge_feat else np.concatenate([g.x for g in dataset], axis=0)
        
        self._mask_nonneg = np.all(feat >= 0, axis=0)
        self._has_nonneg = np.any(self._mask_nonneg)
        if self._has_nonneg:
            self._scaler_nonneg.fit(feat[:, self._mask_nonneg])
        
        self._mask_sign = np.logical_not(self._mask_nonneg)
        self._has_sign = np.any(self._mask_sign)
        if self._has_sign:
            self._scaler_sign.fit(feat[:, self._mask_sign])
        return self
    
    def transform(self, dataset):
        for g in dataset:
            if self._has_nonneg:
                if self.edge_feat:
                    g.e[:, self._mask_nonneg] = \
                        self._scaler_nonneg.transform(g.e[:, self._mask_nonneg])
                else:
                    g.x[:, self._mask_nonneg] = \
                        self._scaler_nonneg.transform(g.x[:, self._mask_nonneg])
            if self._has_sign:
                if self.edge_feat:
                    g.e[:, self._mask_sign] = \
                        self._scaler_sign.transform(g.e[:, self._mask_sign])
                else:
                    g.x[:, self._mask_sign] = \
                        self._scaler_sign.transform(g.x[:, self._mask_sign])
        return dataset

# ----------------------------------------------------------------------------------------------------