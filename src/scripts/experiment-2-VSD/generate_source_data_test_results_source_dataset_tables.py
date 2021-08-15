#!/usr/bin/env python
# coding: utf-8
import pyprojroot
import pandas as pd


ROOT = pyprojroot.here()
RESULTS_ROOT = ROOT.joinpath('results')

VSD_ROOT = RESULTS_ROOT.joinpath('VSD')
TEST_RESULTS_ROOT = VSD_ROOT.joinpath('test_results')
SOURCE_DATA_ROOT = VSD_ROOT.joinpath('source_data')

NET_NAMES = (
    'alexnet',
    'VGG16',
)

SUFFIX_DATASET_TUPLES = (
    ('initialize_CE_largest', 'Pascal VOC'),
    ('transfer_CE_largest', 'ImageNet'),
    ('transfer_searchstims_CE_largest', 'search stimuli')
)


dfs = []

for net_name in NET_NAMES:
    for suffix,  source_dataset in SUFFIX_DATASET_TUPLES:
        test_results_root = TEST_RESULTS_ROOT / f'VSD_{net_name}_{suffix}'
        test_csv_path = sorted(test_results_root.glob('**/*test_results.csv'))
        assert len(test_csv_path) == 1, f'test_csv_path = {test_csv_path}'
        test_csv_path = test_csv_path[0]

        df = pd.read_csv(test_csv_path)
        df['mode'] = 'classify'
        df['source_dataset'] = source_dataset
        dfs.append(df)

all_test_results_df = pd.concat(dfs)
gb = all_test_results_df.groupby(
    ['method', 'source_dataset', 'net_name']
)
df_out = gb.agg(
    acc_largest_mean=pd.NamedAgg('acc_largest', 'mean'),
    acc_largest_std=pd.NamedAgg('acc_largest', 'std'),
)

df_out = df_out.reset_index()
df_out = df_out.sort_values('acc_largest_mean', ascending=False)

df_out.to_csv(
    SOURCE_DATA_ROOT / 'test_results_source_dataset.csv',
    index=False,
)
