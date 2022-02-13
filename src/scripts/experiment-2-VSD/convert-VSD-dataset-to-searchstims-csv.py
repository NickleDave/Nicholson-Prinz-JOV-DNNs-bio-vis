#!/usr/bin/env python
# coding: utf-8
from pathlib import Path

import numpy as np
import pandas as pd
import pyprojroot
import torchvision

import searchnets


VSD_2012_ROOT = Path('/home/bart/Documents/data/voc/VOCdevkit/VOC2012')

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

VOC_CLASS_INT_MAP = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
VOC_INT_CLASS_MAP = dict(zip(range(len(VOC_CLASSES)), VOC_CLASSES))

# use this number of most frequently occurring classes to generate target present / absent csvs
N_TOP_CLASSES = 5


def main():
    vsd_split_csv = pyprojroot.here() / 'data/Visual_Search_Difficulty_v1.0/VSD_dataset_split.csv'
    vsd_df = pd.read_csv(vsd_split_csv)

    vsd_df['root_output_dir'] = str(VSD_2012_ROOT)

    vsd_df['img_file'] = ['JPEGImages/' + file_name + ".jpg" for file_name in vsd_df['img'].values]
    vsd_df['meta_file'] = 'none'  # column added by searchstims, not used

    vsd_df['xml_file'] = ['Annotations/' + file_name + ".xml" for file_name in vsd_df['img'].values]
    target_transform = torchvision.transforms.Compose(
        [searchnets.transforms.ParseVocXml(),
         searchnets.transforms.ClassIntsFromXml()]
    )
    vsd_df['xml_path'] = vsd_df['root_output_dir'] + '/' + vsd_df['xml_file']
    vsd_df['labels'] = vsd_df.xml_path.apply(target_transform)

    vsd_df['stimulus'] = 'VSD'
    vsd_df['set_size'] = vsd_df.labels.apply(len)

    count_records = []
    records = []

    for label in range(20):
        label_as_set = {label}
        df_label = vsd_df[vsd_df.labels.apply(lambda x: bool(label_as_set.intersection(x)))]
        count_records.append(
            {
                'label_int': label,
                'label_str': VOC_INT_CLASS_MAP[label],
                'n_rows': len(df_label)
            }
        )
        set_sizes = df_label.set_size.values
        uniq_set_sizes, counts = np.unique(set_sizes, return_counts=True)
        for set_size, count in zip(uniq_set_sizes, counts):
            records.append(
                {
                    'label_int': label,
                    'label_str': VOC_INT_CLASS_MAP[label],
                    'set_size': set_size,
                    'count': count
                }
            )

    df_label_rows = pd.DataFrame.from_records(count_records)

    these_classes = df_label_rows.sort_values('n_rows', ascending=False).label_int.values[:N_TOP_CLASSES]
    print('will use the following classes:')
    for this_class in these_classes:
        print(VOC_INT_CLASS_MAP[this_class])

    for this_class in these_classes:
        label_str = VOC_INT_CLASS_MAP[this_class]
        df_copy = vsd_df.copy()
        label_as_set = {this_class}
        df_copy['has_label'] = df_copy.labels.apply(lambda x: bool(label_as_set.intersection(x)))
        df_copy['target_condition'] = df_copy['has_label'].map({True: 'present', False: 'absent'})
        csv_fname = f'data/Visual_Search_Difficulty_v1.0/VSD-dataset-split-searchstims-{label_str}.csv'
        csv_path = pyprojroot.here() / csv_fname
        print(
            f'saving: {csv_path}'
        )
        df_copy.to_csv(csv_path)


if __name__ == '__main__':
    main()
