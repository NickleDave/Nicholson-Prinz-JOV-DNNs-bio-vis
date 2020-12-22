#!/usr/bin/env python
# coding: utf-8
from argparse import ArgumentParser

import pandas as pd
import pyprojroot

LOSS_FUNC_ML_TASK_MAP = {
    'CE-largest': 'single-label, largest',
    'CE-random': 'single-label, random',
    'BCE': 'multi-label',
}


def main(source_data_root,
         rm_corr_csv_path,
         test_results_csv_path,
         test_acc_v_r_coeff_csv_filename
         ):
    """generate source data for figure
    that plots accuracy on test set v. r coefficent
    from repeated measures correlations

    Parameters
    ----------
    source_data_root : str, Path
        path to root of directory where "source data" csv files
        that are generated should be saved
    rm_corr_csv_path : str
        path to csv with repeated measures correlations results,
        output of generate_source_data_acc_vsd_corr.py.
        Path should be written relative to source_data_root
    test_results_csv_path : str
        path to csv with results of measuring accuracy on test set,
        output of generate_source_data_test_results.py.
        Path should be written relative to source_data_root
    test_acc_v_r_coeff_csv_filename : str
        filename for .csv that should be saved
        with accuracies and r coefficients combined.
        This is the actual source data used for plotting.
        Saved in source_data_root.
    """
    rm_corr_df = pd.read_csv(
        source_data_root.joinpath(rm_corr_csv_path)
    )

    # get just acc/f1 scores on test set for models trained with transfer learning
    test_results_df = pd.read_csv(source_data_root.joinpath(test_results_csv_path))

    # copy cuz we're going to slice-and-dice
    # to get Dataframe we use for 'x-y' plot comparing test accuracy to r coeff size
    xy_df = rm_corr_df.copy()

    # add colum to rm_corr_df
    xy_df['task (M.L.)'] = xy_df['loss_func'].map(LOSS_FUNC_ML_TASK_MAP)
    # just keep transfer results, now will be same len as test_results_df
    xy_df = xy_df[xy_df.method == 'transfer']
    xy_df['DNN architecture'] = xy_df.net_name.str.replace('_', ' ', regex=False)

    # keep only the columns we need
    COLUMNS_XY = [
        'task (M.L.)', 'DNN architecture', 'loss_func', 'r', 'CI95%', 'dof',  'power', 'pval',
    ]

    xy_df = xy_df[COLUMNS_XY]

    # use test_result_df as index for xy_df, so we can add columns from test_df
    xy_df = xy_df.set_index(['task (M.L.)', 'DNN architecture'])
    test_results_df = test_results_df.set_index(['task (M.L.)', 'DNN architecture'])
    xy_df = xy_df.reindex(index=test_results_df.index)
    for col in ['acc-largest-mean', 'acc-random-mean', 'f1-mean']:
        xy_df[col] = test_results_df[col]
    # finally reset index so we don't lose columns when we convert xy_df to 'long-form'
    xy_df = xy_df.reset_index()

    # make 'long form' so we can use seaborn relplot
    value_vars = ['acc-largest-mean', 'acc-random-mean', 'f1-mean']
    id_vars = [id_var
               for id_var in xy_df.columns.tolist()
               if id_var not in value_vars]
    var_name = 'metric_name'
    value_name = 'metric_val'
    long_test_results_df = pd.melt(xy_df,
                                   id_vars=id_vars,
                                   value_vars=value_vars,
                                   var_name=var_name,
                                   value_name=value_name)

    pairs = [
        ('single-label, largest', 'acc-largest-mean'),
        ('single-label, random', 'acc-random-mean'),
        ('multi-label', 'f1-mean'),
    ]

    long_test_results_df = pd.concat(
        [long_test_results_df[
             (long_test_results_df['task (M.L.)'] == pair[0]) &
             (long_test_results_df['metric_name'] == pair[1])
             ]
         for pair in pairs
         ]
    )
    long_test_results_df.to_csv(source_data_root.joinpath(test_acc_v_r_coeff_csv_filename))


SOURCE_DATA_ROOT = pyprojroot.here().joinpath('results/VSD/source_data')
RM_CORR_CSV_PATH = '8-bins-quantile-strategy/rm_corr.csv'
TEST_RESULTS_CSV_PATH = 'test_results_table_transfer.csv'


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--source_data_root',
                        help=('path to root of directory where "source data" csv files '
                              'that are generated should be saved'),
                        default=SOURCE_DATA_ROOT)
    parser.add_argument('--rm_corr_csv_path',
                        help=('path to csv with repeated measures correlations results, '
                              'output of generate_source_data_acc_vsd_corr.py. '
                              'Path should be written relative to source_data_root'),
                        default=RM_CORR_CSV_PATH)
    parser.add_argument('--test_results_csv_path',
                        help=('path to csv with results of measuring accuracy on test set, '
                              'output of generate_source_data_test_results.py. '
                              'Path should be written relative to source_data_root'),
                        default=TEST_RESULTS_CSV_PATH)
    parser.add_argument('--test_acc_v_r_coeff_csv_filename', default='acc_v_r_coeff.csv',
                        help=('filename for .csv that should be saved '
                              'with accuracies and r coefficients combined. '
                              'This is the actual source data used for plotting. '
                              'Saved in source_data_root.'))
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(source_data_root=args.source_data_root,
         rm_corr_csv_path=args.rm_corr_csv_path,
         test_results_csv_path=args.test_results_csv_path,
         test_acc_v_r_coeff_csv_filename=args.test_acc_v_r_coeff_csv_filename
         )
