#!/usr/bin/env python

import re
import os
import pandas as pd

def get_datasets(dirname):
    return re.findall(r'(KuHar|MotionSense|UCI|RealWorld_waist|RealWorld_thigh)', dirname) 


results_df = pd.DataFrame(columns=['id', 'pretrain DS', 'finetune DS', 'test DS', 'acc'])
results = []

for i, dirname in enumerate(os.listdir('.')):
    if os.path.isdir(dirname) == False:
        continue
    
    datasets = get_datasets(dirname)
    pretrain_ds, finetune_ds, test_ds = datasets[0], datasets[1], datasets[2]

    file = os.path.join(dirname, 'metrics.csv')
    df = pd.read_csv(file)
    acc = df['test_acc'].values[0]

    results.append({'id': i,
                    'pretrain DS':pretrain_ds, 
                    'finetune DS':finetune_ds,
                    'test DS':test_ds,
                    'acc':acc})

results_df = pd.DataFrame(results, columns=['id', 'pretrain DS', 'finetune DS', 'test DS', 'acc'])
results_df.to_csv('results.csv', index=False)

    