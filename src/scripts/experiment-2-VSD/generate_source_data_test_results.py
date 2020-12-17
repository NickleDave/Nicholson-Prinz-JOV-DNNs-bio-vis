#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
from pathlib import Path

import pyprojroot
import pandas as pd


def main(source_data_root,
         test_results_root,
         all_test_results_csv_filename,
         long_test_results_csv_filename):
    """generate .csv files used as source data for figures corresponding to experiments
    carried out with Visual Search Difficulty + PASCAL VOC datasets
    Parameters
    ----------
    source_data_root : str, Path
        path to root of directory where csv files
        that are the source data for figures should be saved.
    test_results_root : str, Path
        path to root of directory that has results.gz files created by `searchnets test` command
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

    # finally, save csvs
    all_test_results_df.to_csv(source_data_root.joinpath(all_test_results_csv_filename), index=False)
    long_test_results_df.to_csv(source_data_root.joinpath(long_test_results_csv_filename), index=False)


ROOT = pyprojroot.here()
RESULTS_ROOT = ROOT.joinpath('results')

VSD_ROOT = RESULTS_ROOT.joinpath('VSD')
TEST_RESULTS_ROOT = VSD_ROOT.joinpath('test_results')
SOURCE_DATA_ROOT = VSD_ROOT.joinpath('source_data')


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--test_results_root',
                        help='path to root of directory that has test results .gz files '
                             'created by searchstims test command',
                        default=TEST_RESULTS_ROOT)
    parser.add_argument('--source_data_root',
                        help=('path to root of directory where "source data" csv files '
                              'that are generated should be saved'),
                        default=SOURCE_DATA_ROOT)
    parser.add_argument('--all_test_results_csv_filename', default='all.csv',
                        help=('filename for .csv that should be saved '
                              'that contains results from **all** results.gz files. '
                              'Saved in source_data_root.'))
    parser.add_argument('--long_test_results_csv_filename', default='acc_diff.csv',
                        help=('''filename for .csv saved that contains all test results, but in "long form", where 
                              'a `metric_name` and `metric_val` column are added, and different metric are all moved
                              to that column ({'acc_largest', 'acc_random', 'f1'}). This "long form" is used for plotting.
                              Saved in source_data_root'''))
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(source_data_root=args.source_data_root,
         all_test_results_csv_filename=args.all_test_results_csv_filename,
         long_test_results_csv_filename=args.long_test_results_csv_filename
         )
