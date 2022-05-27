import os
import sys
import glob
import yaml
import numpy as np
import pandas as pd

if len(sys.argv) > 1:
    results_dir = str(sys.argv[1])
else:
    results_dir = './results/'
    
result_files = [
    fp for fp in glob.glob(os.path.join(results_dir, '*.yaml')) if os.stat(fp).st_size > 0
]    
result_dict = {}
for fp in result_files:
    dataset_name, model = os.path.basename(fp).rstrip('.yaml').split('_')[:2]
    if model not in result_dict:
        result_dict[model] = {}
    with open(fp, 'r') as f:
        result = yaml.load(f, Loader=yaml.SafeLoader)
    for key in ['test']:
        ds_key = f'{dataset_name}.{key}' 
        if ds_key not in result_dict[model]:
            result_dict[model][ds_key] = []
        result_dict[model][ds_key].append(result['results']['test']['acc'])
    
ds_keys = {ds_key for result_dict2 in result_dict.values() for ds_key in result_dict2}


result_dict_mean = {
    model : {
        ds_key : np.mean(result_dict[model][ds_key]) for ds_key in result_dict[model]
    } 
    for model in result_dict
}

result_dict_str = {
    model : {
        ds_key : f'{np.mean(result_dict[model][ds_key]):.3f} +- {np.std(result_dict[model][ds_key]):.3f}' for ds_key in result_dict[model]
    } 
    for model in result_dict
}

for model in result_dict_str:
    for ds_key in ds_keys:
        if ds_key not in result_dict_str[model]:
            result_dict_str[model][ds_key] = '-'

for model in result_dict_str:
    for ds_key in ds_keys:
        if ds_key in result_dict[model]:
            result_dict_str[model][ds_key + '_count'] = len(result_dict[model][ds_key])
        else:
            result_dict_str[model][ds_key + '_count'] = 0

df = pd.DataFrame.from_dict(result_dict_str)

print(df.transpose().sort_index())