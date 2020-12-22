#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pyprojroot
import pandas as pd


def _t1_summary(series, n_decimals=3):
    """adapted from tableone package

    https://github.com/tompollard/tableone/blob/5b3d64426e0d6cc9877fe38690fed190510b2c74/tableone/tableone.py#L814
    """
    f = '{{:.{}f}} ({{:.{}f}})'.format(n_decimals, n_decimals)
    return f.format(np.nanmean(series.values), np.nanstd(series.values))


LOSS_FUNC_CATEGORY_MAP = {
    'CE-largest': 0,
    'CE-random': 1,
    'BCE': 2,
}

LOSS_FUNC_CATEGORY_ML_TASK_MAP = {
    0: 'single-label, largest',
    1: 'single-label, random',
    2: 'multi-label',
}


def test_results_table(all_test_results_df,
                       method='transfer',
                       groupby=['loss_func_category', 'net_name'],
                       agg={'acc_largest': _t1_summary, 'acc_random': _t1_summary, 'f1': _t1_summary},
                       sort_values=['loss_func_category', 'acc_largest'],
                       columns=['acc. (largest object)', 'acc. (random object)', 'f1 (macro)'],
                       ascending=(True, True),
                       ):
    df_test_table = all_test_results_df[all_test_results_df.method == method]
    df_test_table['loss_func_category'] = df_test_table['loss_func'].map(LOSS_FUNC_CATEGORY_MAP)
    df_test_table = df_test_table.groupby(groupby).agg(agg).sort_values(sort_values, ascending=ascending)

    # rename levels for loss functions category
    df_test_table.index = df_test_table.index.set_levels(
        [LOSS_FUNC_CATEGORY_ML_TASK_MAP[lbl] for lbl in df_test_table.index.levels[0]],
        level=0)

    df_test_table.index = df_test_table.index.set_levels(
        [lbl.replace('_', ' ') for lbl in df_test_table.index.levels[1]],
        level=1)

    df_test_table.index = df_test_table.index.rename(names=['task (M.L.)', 'DNN architecture'])

    df_test_table.columns = columns
    return df_test_table


def main(test_results_root,
         source_data_root,
         all_test_results_csv_filename,
         long_test_results_csv_filename):
    """generate .csv files used as source data for tables / figures
    that report model performance on test set
    in experiments carried out with Visual Search Difficulty + PASCAL VOC datasets

    Parameters
    ----------
    test_results_root : str, Path
        path to root of directory that has test_results.csv files created by `searchnets test` command
    source_data_root : str, Path
        path to root of directory where csv files
        that are the source data for figures should be saved.
    all_test_results_csv_filename : str
        filename for .csv saved that contains results from **all** results.gz files.
        Saved in source_data_root.
    long_test_results_csv_filename : str
        filename for .csv saved that contains all test results, but in "long form", where 
        a 'metric_name' and 'metric_val' column are added, and different metric are all moved 
        to that column ({'acc_largest', 'acc_random', 'f1'}). This "long form" is used for plotting.
        Saved in source_data_root.
    """
    test_results_root = Path(test_results_root)
    test_csv_paths = sorted(test_results_root.glob('**/*test_results.csv'))

    dfs = []
    for test_csv_path in test_csv_paths:
        df = pd.read_csv(test_csv_path)
        df['mode'] = 'classify'
        dfs.append(df)

    all_test_results_df = pd.concat(dfs)

    # "melt" so that metrics are rows instead of columns, makes plotting more convenient
    # this adds a 'metric_name' column where name is one of {'acc_largest', 'acc_random', 'f1'}
    # and then a column 'metric_val' with the value computed corresponding to 'metric_name'
    value_vars = ['acc_largest', 'acc_random', 'f1']
    id_vars = [id_var 
               for id_var in all_test_results_df.columns.tolist() 
               if id_var not in value_vars]
    var_name = 'metric_name'
    value_name = 'metric_val'
    long_test_results_df = pd.melt(all_test_results_df,
                                   id_vars=id_vars,
                                   value_vars=value_vars,
                                   var_name=var_name,
                                   value_name=value_name)

    # realize after writing the script I need mean and std in separate columns
    # so I can more easily plot mean test accuracy v. r values from correlation.
    # Create those here first.
    agg = {k: ['mean', 'std'] for k in ['acc_largest', 'acc_random', 'f1']}
    columns = [
        'acc-largest-mean',
        'acc-largest-std',
        'acc-random-mean',
        'acc-random-std',
        'f1-mean',
        'f1-std',
    ]
    sort_values=['loss_func_category', ('acc_largest', 'mean')]
    df_test_table_transfer = test_results_table(all_test_results_df,
                                                method='transfer',
                                                agg=agg,
                                                sort_values=sort_values,
                                                columns=columns)
    df_test_table_initialize = test_results_table(all_test_results_df,
                                                  method='initialize',
                                                  agg=agg,
                                                  sort_values=sort_values,
                                                  columns=columns)

    df_test_table_transfer_mean_sd_single_col = test_results_table(all_test_results_df, method='transfer')
    df_test_table_initialize_mean_sd_single_col = test_results_table(all_test_results_df, method='initialize')

    # finally, save csvs
    all_test_results_df.to_csv(source_data_root.joinpath(all_test_results_csv_filename), index=False)
    long_test_results_df.to_csv(source_data_root.joinpath(long_test_results_csv_filename), index=False)

    df_test_table_transfer.to_csv(source_data_root.joinpath('test_results_table_transfer.csv'))
    df_test_table_initialize.to_csv(source_data_root.joinpath('test_results_table_initialize.csv'))
    df_test_table_transfer_mean_sd_single_col.to_csv(source_data_root.joinpath(
        'test_results_table_transfer_mean_sd_single_col.csv'
    ))
    df_test_table_initialize_mean_sd_single_col.to_csv(source_data_root.joinpath(
        'test_results_table_initialize_mean_sd_single_col.csv'
    ))
    # also save tables as Excel files, to import into Word
    df_test_table_transfer_mean_sd_single_col.to_excel(source_data_root.joinpath(
        'test_results_table_transfer_mean_sd_single_col.xlsx'
    ))
    df_test_table_initialize_mean_sd_single_col.to_excel(source_data_root.joinpath(
        'test_results_table_initialize_mean_sd_single_col.xlsx'
    ))


ROOT = pyprojroot.here()
RESULTS_ROOT = ROOT.joinpath('results')

VSD_ROOT = RESULTS_ROOT.joinpath('VSD')
TEST_RESULTS_ROOT = VSD_ROOT.joinpath('test_results')
SOURCE_DATA_ROOT = VSD_ROOT.joinpath('source_data')


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--test_results_root',
                        help='path to root of directory that has test results .csv files '
                             'created by searchnets test command',
                        default=TEST_RESULTS_ROOT)
    parser.add_argument('--source_data_root',
                        help=('path to root of directory where "source data" csv files '
                              'that are generated should be saved'),
                        default=SOURCE_DATA_ROOT)
    parser.add_argument('--all_test_results_csv_filename', default='all_test_results.csv',
                        help=('filename for .csv that should be saved '
                              'that contains results from **all** results.gz files. '
                              'Saved in source_data_root.'))
    parser.add_argument('--long_test_results_csv_filename', default='all_test_results_long_form.csv',
                        help=('''filename for .csv saved that contains all test results, but in "long form", where 
                              'a `metric_name` and `metric_val` column are added, and different metric are all moved
                              to that column ({'acc_largest', 'acc_random', 'f1'}). 
                              This "long form" is used for plotting.
                              Saved in source_data_root'''))
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(test_results_root=args.test_results_root,
         source_data_root=args.source_data_root,
         all_test_results_csv_filename=args.all_test_results_csv_filename,
         long_test_results_csv_filename=args.long_test_results_csv_filename
         )
