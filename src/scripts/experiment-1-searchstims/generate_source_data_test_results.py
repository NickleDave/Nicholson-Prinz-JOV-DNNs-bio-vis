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
    return f.format(np.nanmean(series.values), np.nanmean(series.values))


METHOD_CATEGORY_MAP = {
    'transfer': 0,
    'initialize': 1,
}

METHOD_CATEGORY_ML_TASK_MAP = {
    0: 'transfer learning',
    1: 'randomly-initialized weights',
}


def test_results_table(df_test_table,
                       groupby=['method_category', 'net_name'],
                       agg={'acc': _t1_summary},
                       sort_values=['method_category', 'acc'],
                       columns=['acc. Mean (S.D.)'],
                       ascending=(True, True),
                       ):
    df_test_table['method_category'] = df_test_table['method'].map(METHOD_CATEGORY_MAP)
    df_test_table = df_test_table.groupby(groupby).agg(agg).sort_values(sort_values, ascending=ascending)

    # rename levels for loss functions category
    df_test_table.index = df_test_table.index.set_levels(
        [METHOD_CATEGORY_ML_TASK_MAP[lbl] for lbl in df_test_table.index.levels[0]],
        level=0)

    df_test_table.index = df_test_table.index.set_levels(
        [lbl.replace('_', ' ') for lbl in df_test_table.index.levels[1]],
        level=1)

    df_test_table.index = df_test_table.index.rename(names=['training method', 'DNN architecture'])

    df_test_table.columns = columns
    return df_test_table


def main(test_results_root,
         source_data_root,
         all_test_results_csv_filename,
):
    """generate .csv files used as source data for tables / figures
    that report model performance on test set
    in experiments carried out with 'searchstims' datasets

    Parameters
    ----------
    test_results_root : str, Path
        path to root of directory that has results.gz files created by `searchnets test` command
    source_data_root : str, Path
        path to root of directory where csv files
        that are the source data for figures should be saved.
    all_test_results_csv_filename : str
        filename for .csv saved that contains results from **all** results.gz files.
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

    # realize after writing the script I need mean and std in separate columns
    # so I can more easily plot mean test accuracy v. r values from correlation.
    # Create those here first.
    agg = {'acc': ['mean', 'std']}
    columns = [
        'acc-mean',
        'acc-std',
    ]
    sort_values=['method_category', ('acc', 'mean')]
    df_test_table = test_results_table(all_test_results_df,
                                       agg=agg,
                                       sort_values=sort_values,
                                       columns=columns)

    df_test_table_mean_sd_single_col = test_results_table(all_test_results_df)

    # finally, save csvs
    all_test_results_df.to_csv(source_data_root.joinpath(all_test_results_csv_filename), index=False)

    df_test_table.to_csv(source_data_root.joinpath('test_results_table.csv'))
    df_test_table_mean_sd_single_col.to_csv(source_data_root.joinpath(
        'test_results_table_mean_sd_single_col.csv'
    ))

    # also save table as Excel file, to import into Word
    df_test_table_mean_sd_single_col.to_excel(source_data_root.joinpath(
        'test_results_table_mean_sd_single_col.xlsx'
    ))


ROOT = pyprojroot.here()
RESULTS_ROOT = ROOT.joinpath('results')

SEARCHSTIMS_ROOT = RESULTS_ROOT.joinpath('searchstims')
TEST_RESULTS_ROOT = SEARCHSTIMS_ROOT.joinpath('results_gz/3stims')
SOURCE_DATA_ROOT = SEARCHSTIMS_ROOT.joinpath('source_data')


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
                              'that contains results from **all** test_results.csv files. '
                              'Saved in source_data_root.'))
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(test_results_root=args.test_results_root,
         source_data_root=args.source_data_root,
         all_test_results_csv_filename=args.all_test_results_csv_filename,
         )
