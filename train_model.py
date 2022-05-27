import numpy as np
import tensorflow as tf

from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.optimizers.schedules import CosineDecay

from nn_ops.models import GNNModel

from utils import EarlyStoppingAlwaysRestore

# ----------------------------------------------------------------------------------------------------

def train_model(dataset, config={}, verbose=0):
    
    loader_tr, loader_va, loader_te = dataset.loaders
    
    n_epochs = config['opt']['n_epochs'] if 'n_epochs' in config['opt'] else 200
    lr_init = config['opt']['lr_init'] if 'lr_init' in config['opt'] else 1e-3
    lr = lr_init
    
    lr_schedule = config['opt']['lr_schedule'] \
        if 'lr_schedule' in config['opt'] else None
    
    cbs = []
    monitor = config['opt']['monitor'] if 'monitor' in config['opt'] else 'loss'
    mode = config['opt']['mode'] if 'mode' in config['opt'] else 'min'
    
    if lr_schedule == 'plateau':
        lr_patience = config['opt']['lr_patience'] \
            if 'lr_patience' in config['opt'] else 10
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor=monitor, mode=mode, patience=lr_patience, factor=0.5, min_lr=1e-5, min_delta=0.,
            verbose=verbose)
        cbs.append(reduce_lr)
    elif lr_schedule == 'cosine':
        lr = CosineDecay(lr_init, n_epochs * loader_tr.steps_per_epoch)
        
    opt = optimizers.Adam(lr)        
    
    if 'es_patience' in config['opt']:
        es_patience = config['opt']['es_patience'] or n_epochs
        es = EarlyStoppingAlwaysRestore(
            monitor=monitor, mode=mode, patience=es_patience, min_delta=0., verbose=verbose,
            restore_best_weights=True)
        cbs.append(es)
        
    model = GNNModel(dataset.n_labels, node_level=dataset.node_level, single_mode=dataset.single_mode,
                     **config['model'])

    
    model.compile(opt, dataset.loss, weighted_metrics=dataset.metrics)
    
    hist = model.fit(loader_tr.load(), epochs=n_epochs, steps_per_epoch=loader_tr.steps_per_epoch,
                     validation_data=loader_va.load(), validation_steps=loader_va.steps_per_epoch,
                     callbacks=cbs, verbose=verbose)
    
    scores = [model.evaluate(loader.load(), steps=loader.steps_per_epoch)
              for loader in (loader_tr, loader_va, loader_te)]
    results = {split :
               {
                   score_type : scores[i][j]
                   for j, score_type in enumerate(['loss'] + [met.name for met in dataset.metrics])
               }
               for i, split in enumerate(['train', 'val', 'test'])
              }
    return results
    
# ----------------------------------------------------------------------------------------------------