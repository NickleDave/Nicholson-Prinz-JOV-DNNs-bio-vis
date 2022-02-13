#!/usr/bin/env python
# coding: utf-8
"""generates final source data for figure that plots
accuracy vs set size for
models trained on Visual Search Difficulty / Pascal VOC dataset.

Five different targets
and four different architectures

run this after running 'generate-source-data-csv.sh'
"""
import pandas as pd
import pyprojroot


SOURCE_DATA_ROOT = pyprojroot.here('results/VSD/source_data')

TARGETS = [
    'car',
    'cat',
    'dog',
    'chair',
    'person'
]


MAX_SET_SIZE = 10

df_all = []

for target in TARGETS:
    df = pd.read_csv(
        SOURCE_DATA_ROOT.joinpath(f'{target}/all.csv')
    )
    df = df.sort_values(
        ['net_name', 'method', 'mode', 'learning_rate', 'net_number', 'stimulus', 'set_size',]
    )
    df = df[df.target_condition == 'both']
    df = df[df.set_size < MAX_SET_SIZE]
    df['stimulus'] = target

    df_all.append(df)
    
df_all = pd.concat(df_all)
del df

df_all.to_csv(
    SOURCE_DATA_ROOT / 'VSD-searchstims-acc-v-set-size.csv'
)
