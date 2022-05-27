from abc import ABCMeta, abstractmethod

class Dataset(metaclass=ABCMeta):
    @abstractmethod
    def split(self):
        pass
    
    @abstractmethod
    def preprocess(self):
        pass
    
    @property
    def n_labels(self):
        return self.dataset[0].y.shape[-1]
    
    @property
    def single_mode(self):
        return len(self.dataset) == 1
    
    @property
    def node_level(self):
        return self.dataset[0].y.ndim == 2
    
    @property
    @abstractmethod
    def loaders(self):
        pass
    
    @property
    @abstractmethod
    def loss(self):
        pass
    
    @property
    @abstractmethod
    def metrics(self):
        pass