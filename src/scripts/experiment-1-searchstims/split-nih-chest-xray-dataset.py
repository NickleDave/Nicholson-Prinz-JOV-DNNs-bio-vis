#!/usr/bin/env python
# coding: utf-8
"""
This script splits the NIH Chest X-Ray Dataset from Kaggle
(https://www.kaggle.com/nih-chest-xrays/data)
into two classes of equal size, "Finding" and "No Finding",
and three splits: "train", "val", and "test".
The goal is to train and evaluate CNNs on a binary classification problem.

---- BEFORE RUNNING THIS SCRIPT ---
- move all images from the original dataset into a single directory, './images'
- make directories 'train', 'val' and 'test' in the root of the directory, at the same level as './images'
"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import sklearn.model_selection
from tqdm import tqdm


XRAY_ROOT = Path('/home/bart/Documents/data/nih-chest-xray/')

# #### get 'train/val' and 'test' lists that we will use to add a 'split' column to DataFrame
with XRAY_ROOT.joinpath('train_val_list.txt').open('r') as fp:
    train_val_list = fp.read().splitlines()

with XRAY_ROOT.joinpath('test_list.txt').open('r') as fp:
    test_list = fp.read().splitlines()


# #### load dataset csv into DataFrame
df = pd.read_csv(XRAY_ROOT / 'Data_Entry_2017.csv')

# add 'split' column
df['isin_trainval'] = df['Image Index'].isin(train_val_list)
df['split'] = df['isin_trainval'].map({True: 'trainval', False: 'test'})
if not df[df['split'] == 'test']['Image Index'].isin(test_list).all():
    raise ValueError(
        'some rows in test split are not in test list loaded from .txt file'
    )

# add 'label' column where label is either 'no-finding' or 'yes-finding' (binary classification)
def finding_or_no(lbl):
    return 'no-finding' if lbl == 'No Finding' else 'yes-finding'

df['label'] = df['Finding Labels'].apply(func=finding_or_no)

# split off a validation set
# first get train and test set so we can just get val from train set
df_test = df[df['split'] == 'test']
df_trainval = df[df['split'] == 'trainval']

VAL_SIZE = 0.1

df_train, df_val = sklearn.model_selection.train_test_split(df_trainval, test_size=VAL_SIZE, random_state=42, shuffle=True)

for df, split in zip(
    (df_train, df_val, df_test),
    ('train', 'val', 'test')
):
    print(f'split: {split}, df len: {len(df)}')
    print(f'n. samples "yes-finding": {len(df[df.label == "yes-finding"])}')
    print(f'n. samples "no-finding": {len(df[df.label == "no-finding"])}')

# ---- finally, move all files from images/ into split directories
for df, split in zip(
    (df_train, df_val, df_test),
    ('train', 'val', 'test')
):
    for label in ('yes-finding', 'no-finding'):
        print (f'moving images for split {split}, class {label}')
        df_label = df[df.label == label]
        for img_filename in tqdm(df_label['Image Index'].values):
            src = XRAY_ROOT / 'images' / img_filename
            assert src.exists()
            src = str(src)
            dst = str(XRAY_ROOT / split / label)
            shutil.move(src, dst)
