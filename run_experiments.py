import os
import sys
if len(sys.argv) > 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys.argv[1])
    
from copy import deepcopy
    
import yaml
import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    pass

tf.config.experimental.enable_op_determinism()

from train_model import train_model
from datasets import load_dataset


save_dir = './results/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

with open('./configs.yaml', 'r') as f:
    configs = yaml.load(f, Loader=yaml.SafeLoader)

for cfg_name in configs:
    config = deepcopy(configs[cfg_name])
    n_splits, n_trials = config['experiment']['n_splits'], config['experiment']['n_trials']
    ds_names = config['data']['dataset_name'].split(',')
    for ds_name in ds_names:
        data_config = config['data'].copy()
        data_config['dataset_name'] = ds_name
        train_config = {'model' : config['model'].copy(), 'opt' : config['opt'].copy()}
        for split_id in range(n_splits):
            data_config['split_id'] = split_id
            for trial_id in range(n_trials):
                seed = split_id * n_trials + trial_id
                fname = f'{ds_name}_{cfg_name}_{split_id}.yaml' \
                    if n_trials==1 else f'{ds_name}_{cfg_name}_{split_id}_{trial_id}.yaml'
                fp = os.path.join(save_dir, fname)
                if os.path.exists(fp):
                    print(f'{fp} exists, skipping...')
                    continue
                with open(fp, 'w') as f:
                    pass
                tf.keras.utils.set_random_seed(seed)
                dataset = load_dataset(**data_config).split().preprocess()
                results = train_model(dataset, config=config, verbose=1)
                config['results'] = results
                with open(fp, 'w') as f:
                    yaml.dump(config, f)
