#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import pyprojroot
from sklearn.preprocessing import KBinsDiscretizer
from statsmodels.formula.api import ols


def main(test_results_root,
         source_data_root,
         accuracy_csv_filename,
         rm_corr_csv_filename,
         n_bins=8,
         strategy='quantile',
         ):
    """

    Parameters
    ----------
    test_results_root : str, Path
        path to root of directory that has results.gz files created by `searchnets test` command
    source_data_root : str, Path
        path to root of directory where csv files
        that are the source data for figures should be saved.
    accuracy_csv_filename : str, Path
    rm_corr_csv_filename : str, Path
    n_bins : int
        number of bins to use with KBinsDiscretizer
        when binning Visual Search Difficulty scores.
        Default is 8.
    strategy : str
        strategy to use with KBinsDiscretizer
        when binning Visual Search Difficulty scores.
        One of {'uniform', 'quantile'}.
        Default is 'quantile'.
    """
    test_results_root = Path(test_results_root)
    source_data_root = Path(source_data_root)

    # ## get all the `assay_images` csvs from each model / net / mode / method, concatenate
    assay_images_csvs = sorted(test_results_root.glob('**/*assay_images.csv'))

    # before loading all csvs:
    # * load one and discretize visual search difficulty scores
    #   - we choose the bins using the first csv, all will be the same (it's always the same test set)
    #     + choose **before** filtering further
    #       because we want bins to capture the entire range of scores
    #       - e.g. if we bin after filtering to keep only images with single items, this will change the range
    #         and the width of the bins
    #     + we choose bins so that each has same # of samples,
    #       instead of having equal width bins
    first_csv_df = pd.read_csv(assay_images_csvs[0])

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    vsd_score_bin = discretizer.fit_transform(first_csv_df['vsd_score'].values.reshape(-1, 1))
    vsd_score_bin = pd.Series(vsd_score_bin.ravel()).astype('category')

    # now we load all `assay_images` csvs
    # and add `vsd_score_bin` column that we just made to each one
    df_list = []

    for assay_images_csv in assay_images_csvs:
        df = pd.read_csv(assay_images_csv)
        df['vsd_score_bin'] = vsd_score_bin
        df_list.append(df)

    assay_images_df = pd.concat(
        df_list
    )

    # declare column order, just for tidyness
    COLUMNS = [
        'net_name',
        'replicate',
        'mode',
        'method',
        'loss_func',
        'TP',
        'TN',
        'FN',
        'FP',
        'vsd_score',
        'vsd_score_bin',
        'n_items',
        'img_name',
        'img_path',
        'restore_path',
        'voc_test_index',
    ]
    assay_images_df = assay_images_df[COLUMNS]

    # keep only 'classify' mode, not 'detect'
    assay_images_df = assay_images_df[assay_images_df['mode'] == 'classify']

    # ## use `groupby` to compute accuracy for each experimental level,
    # including the visual search difficulty score (binned)
    # ### declare labels to group by experiment levels **and** visual search difficulty score (bin)
    # (used multiple times below)
    GROUP_LABELS = ['net_name', 'replicate', 'mode', 'method', 'loss_func', 'vsd_score_bin']

    # ### measure accuracy as a function of (binned) visual search difficulty scores
    # #### first for just cases where there is only 1 item in image
    # - filter by loss function: we only want single-label classification, so remove 'BCE'
    # - also filter by number of objects / items, we only want images with one object present
    single_label_df = assay_images_df[assay_images_df.loss_func.isin(['CE-largest', 'CE-random'])]
    single_label_df = single_label_df[single_label_df['n_items'] == 1]

    # - then split into groups that correspond to experimental levels + variables of interest
    single_label_gb = single_label_df.groupby(GROUP_LABELS)
    single_label_groups = single_label_gb.groups

    # * finally create new dataframe where each experimental level has a corresponding accuracy measure
    def acc(df):
        assert np.all(df['TP'].isin([0, 1])), "not all true positive values were zero or one"
        return df['TP'].sum() / len(df)

    single_label_records = defaultdict(list)

    for a_group in sorted(single_label_groups.keys()):
        a_group_df = single_label_gb.get_group(a_group)

        for grp_label, grp_val in zip(GROUP_LABELS, a_group):
            single_label_records[grp_label].append(grp_val)

        # treat each image as a 'trial'
        single_label_records['n_trials'].append(len(a_group_df))

        single_label_records['acc'].append(
            acc(a_group_df)
        )

    single_label_acc_df = pd.DataFrame.from_records(single_label_records)

    # ### declare COLUMNS for accuracy `DataFrame`s -- will use the same columns for multi-label accuracy
    ACC_COLUMNS = [
        'method',
        'mode',
        'loss_func',
        'net_name',
        'replicate',
        'vsd_score_bin',
        'acc',
        'n_trials',
    ]

    single_label_acc_df = single_label_acc_df[ACC_COLUMNS]

    # #### now measure accuracy for images with any number of objects,
    # using networks trained for multi-label classification
    # - filter by loss function: we want multi-label classification, so keep only 'BCE'
    multi_label_df = assay_images_df[assay_images_df.loss_func.isin(['BCE'])]
    multi_label_gb = multi_label_df.groupby(GROUP_LABELS)
    multi_label_groups = multi_label_gb.groups

    def multi_label_acc(df):
        acc_series = df['TP'] / (df['TP'] + df['FN'])
        return acc_series.mean()

    multi_label_records = defaultdict(list)

    for a_group in sorted(multi_label_groups.keys()):
        a_group_df = multi_label_gb.get_group(a_group)

        for grp_label, grp_val in zip(GROUP_LABELS, a_group):
            multi_label_records[grp_label].append(grp_val)

        # treat each image as a 'trial'
        multi_label_records['n_trials'].append(len(a_group_df))

        multi_label_records['acc'].append(
            multi_label_acc(a_group_df)
        )

    multi_label_acc_df = pd.DataFrame.from_records(multi_label_records)
    multi_label_acc_df = multi_label_acc_df[ACC_COLUMNS]

    # #### concatenate single-label and multi-label accuracy dataframe
    # * 'loss_func' column ('CE-random' and 'BCE') is a proxy for 'single-label' and 'multi-label'
    # * loop through the concatenated dataframe to run regression,
    #   and to get predictions from that linear model for plotting
    acc_df = pd.concat((single_label_acc_df, multi_label_acc_df))

    def rm_corr_for_plot(data, x, y, subject):
        """run regression on each replicate,
        just to get predicted y values of that linear model
        that we can use to plot

        adapted from pingouin.plot_rm_corr
        https://github.com/raphaelvallat/pingouin/blob/08560bb978949e97653b203f7374e90b022b88fb/pingouin/plotting.py#L851
        """
        # Fit ANCOVA model
        # https://patsy.readthedocs.io/en/latest/builtins-reference.html
        # C marks the data as categorical
        # Q allows to quote variable that do not meet Python variable name rule
        # e.g. if variable is "weight.in.kg" or "2A"
        formula = "Q('%s') ~ C(Q('%s')) + Q('%s')" % (y, subject, x)
        model = ols(formula, data=data).fit()

        # Fitted values
        return model.fittedvalues

    rm_corr_records = defaultdict(list)
    acc_df['pred'] = 0.  # dummy value we replace in loop below

    for mode in ['classify']:
        for method in ['transfer', 'initialize']:
            for row, loss_func in enumerate(['CE-random', 'CE-largest', 'BCE']):
                for col, net_name in enumerate(acc_df['net_name'].unique()):
                    sub_df = acc_df[
                        (acc_df['mode'] == mode) &
                        (acc_df['method'] == method) &
                        (acc_df['loss_func'] == loss_func) &
                        (acc_df['net_name'] == net_name)
                    ]
                    if len(sub_df) == 0:
                        continue

                    rm_corr_df = pg.rm_corr(data=sub_df, x='vsd_score_bin', y='acc', subject='replicate')
                    for k, v in rm_corr_df.to_dict(orient='records')[0].items():  # records will be one-item list
                        rm_corr_records[k].append(v)
                    for k, v in zip(('mode', 'method', 'loss_func', 'net_name'), (mode, method, loss_func, net_name)):
                        rm_corr_records[k].append(v)

                    pred = rm_corr_for_plot(data=sub_df, x='vsd_score_bin', y='acc', subject='replicate')
                    # add predicted values from regression to dataframe, to use for plotting later
                    acc_df.loc[
                        ((acc_df['mode'] == mode) &
                         (acc_df['method'] == method) &
                         (acc_df['loss_func'] == loss_func) &
                         (acc_df['net_name'] == net_name)),
                        'pred'] = pred

    # #### save dataframe of repeated measures correlation results
    rm_corr_df = pd.DataFrame.from_records(rm_corr_records)

    # save results in a directory inside source data root named "number of bins + binning strategy"
    out_dir = source_data_root.joinpath(
        f'{n_bins}-bins-{strategy}-strategy'
    )
    out_dir.mkdir(exist_ok=True, parents=False)
    # finally, save csvs + bin edges
    acc_df.to_csv(out_dir.joinpath(accuracy_csv_filename), index=False)
    rm_corr_df.to_csv(out_dir.joinpath(rm_corr_csv_filename), index=False)
    np.savetxt(
        fname=out_dir.joinpath('bin_edges.np.txt'),
        X=discretizer.bin_edges_[0]
    )


# constants
ROOT = pyprojroot.here()
RESULTS_ROOT = ROOT.joinpath('results')

VSD_ROOT = RESULTS_ROOT.joinpath('VSD')
TEST_RESULTS_ROOT = VSD_ROOT.joinpath('test_results')
SOURCE_DATA_ROOT = VSD_ROOT.joinpath('source_data')

N_BINS = 8


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
    parser.add_argument('--n_bins', type=int,
                        help=('number of bins to use with KBinsDiscretizer'
                              'when binning Visual Search Difficulty scores'),
                        default=N_BINS)
    parser.add_argument('--strategy',
                        help=('''"strategy" to use with KBinsDiscretizer
                              when binning Visual Search Difficulty scores.
                              One of {'uniform','quantile'}. Default is 'quantile'.'''),
                        choices=('uniform', 'quantile'),
                        default='quantile')
    parser.add_argument('--accuracy_csv_filename', default='acc.csv',
                        help=('filename for .csv that should be saved '
                              'that contains all accuracies computed per group.'
                              'Saved in source_data_root.'))
    parser.add_argument('--rm_corr_csv_filename', default='rm_corr.csv',
                        help=('''filename for .csv saved that contains 
                              repeated measures correlation results.
                              Saved in source_data_root'''))
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(test_results_root=args.test_results_root,
         source_data_root=args.source_data_root,
         n_bins=args.n_bins,
         strategy=args.strategy,
         accuracy_csv_filename=args.accuracy_csv_filename,
         rm_corr_csv_filename=args.rm_corr_csv_filename
         )
