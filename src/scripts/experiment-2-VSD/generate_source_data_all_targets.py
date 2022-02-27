"""
generate source data .csv for figure
that combines results across all experiments
with Pascal VOC dataset, treating
different classes as candidate targets
"""
import pandas as pd
import pyprojroot

SOURCE_DATA_ROOT = pyprojroot.here('results/VSD/source_data/')

NET_NAMES = [
    'alexnet',
    'VGG16',
    'CORnet_Z',
    'CORnet_S',
]

TARGET_CLASSES = [
    'person',
    'cat',
    'dog',
    'chair',
    'car',
]

SET_SIZES = list(range(6))

df_all = []

for target_class in TARGET_CLASSES:
    df = pd.read_csv(SOURCE_DATA_ROOT / target_class / 'all.csv')
    df['stimulus'] = target_class  # HACK
    df = df[
        (df.net_name.isin(NET_NAMES)) &
        (df.method == 'transfer') &
        (df.set_size.isin(SET_SIZES))
        ]
    df_all.append(df)

df_all = pd.concat(df_all)

df_all = df_all[df_all.target_condition == 'both']

# some of these columns are expected by figure plotting code,
# kept for that reason
df_all = df_all[
    ['method', 'net_name', 'net_number', 'stimulus', 'target_condition',
     'set_size', 'n_correct', 'n_trials', 'accuracy']
]

df_all = df_all.sort_values(by=['net_name', 'net_number', 'stimulus', 'set_size',])

# map set sizes to consecutive integers
set_size_uniq = df_all.set_size.unique()
set_size_map = dict(zip(set_size_uniq, range(len(set_size_uniq))))
df_all['set_size_int'] = df_all.set_size.map(set_size_map)
df_all.accuracy = df_all.accuracy * 100

df_all.to_csv(SOURCE_DATA_ROOT / 'VSD-searchstims-all.csv', index=False)
