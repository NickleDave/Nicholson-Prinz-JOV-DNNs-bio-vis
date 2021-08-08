#!/usr/bin/env python
# coding: utf-8
"""generate source data for figures showing fits to psychometric functions
and derived analysis from those fits"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyprojroot
import scipy.stats
from scipy.optimize import curve_fit


D_RANGE = np.linspace(0, 1, 1000)


def psychometric_func(x,
                      alpha,
                      beta,
                      chance=0.5):
    """psychometric function"""
    return chance + (1-chance) / (1 + np.exp(-(x - alpha) / beta))


def fits_df_from_discrim_df(discrim_df,
                            d_range=D_RANGE,
                            par0=np.asarray([0.5, 0.05]),
                            threshold_acc=0.75):
    records = []  # list of dict we will convert to a pandas.DataFrame

    for net_name in discrim_df.net_name.unique():
        for method_source_dataset in discrim_df.method_source_dataset.unique():
            this_df = discrim_df[
                (discrim_df.net_name == net_name) &
                (discrim_df.method_source_dataset == method_source_dataset)
            ]
            for subject in this_df.net_number.unique():
                df_subject = this_df[this_df.net_number == subject]

                for set_size in df_subject.set_size.unique():
                    df_set_size = df_subject[df_subject.set_size == set_size]
                    df_set_size = df_set_size.sort_values(by='discrim_pct')
                    discrim_pct = df_set_size.discrim_pct.values
                    acc = df_set_size.accuracy.values

                    try:
                        popt, pcov = curve_fit(psychometric_func, discrim_pct, acc, par0)
                        params_cov = np.diag(pcov)
                    except RuntimeError:
                        print(
                            f'fitting failed for {net_name}, {method_source_dataset}, net number {subject}, set size {set_size}'
                        )
                        popt = params_cov = np.array([np.nan, np.nan])

                    y = psychometric_func(d_range, popt[0], popt[1])
                    # where (on x-axis) did subject have threshold performance for this set size?
                    th_ind = np.argmin(np.abs(y - threshold_acc))
                    discrim_threshold = d_range[th_ind]

                    records.append(
                        {
                            'net_name': net_name,
                            'method': this_df.method.unique()[0],
                            'source_dataset': this_df.source_dataset.unique()[0],
                            'net_number': subject,
                            'set_size': set_size,
                            'alpha': popt[0],
                            'alpha_cov': params_cov[0],
                            'beta': popt[1],
                            'beta_cov': params_cov[1],
                            'threshold_acc': threshold_acc,
                            'discrim_threshold': discrim_threshold,
                        }
                    )

    fits_df = pd.DataFrame.from_records(records)
    return fits_df


def regress_net_number(fits_df):
    """do regression to get a slope for each "subject",
    i.e. training replicate, 'net_number',
    from a DataFrame"""
    slopes_records = []

    for net_name in fits_df.net_name.unique():
        for method in fits_df.method.unique():
            for source_dataset in fits_df.source_dataset.unique():
                this_fits_df = fits_df[
                    (fits_df.net_name == net_name) &
                    (fits_df.method == method) &
                    (fits_df.source_dataset == source_dataset)
                    ]
                if len(this_fits_df) == 0:
                    continue
                for net_number in this_fits_df.net_number.unique():
                    netnum_df = this_fits_df[this_fits_df.net_number == net_number]
                    x, y = netnum_df.log_set_size.values, netnum_df.log_discrim_threshold.values
                    result = scipy.stats.linregress(x, y)
                    record = {
                        'net_name': net_name,
                        'method': method,
                        'source_dataset': source_dataset,
                        'net_number': net_number,
                        'intercept': result.intercept,
                        'intercept_stderr': result.intercept_stderr,
                        'pvalue': result.pvalue,
                        'rvalue': result.rvalue,
                        'slope': result.slope,
                        'stderr': result.stderr,
                    }
                    slopes_records.append(record)

    slopes_df = pd.DataFrame.from_records(slopes_records)
    return slopes_df


SOURCE_DATASETS_TO_USE = (
    'None',  # initialized and trained just to classify search stimuli as target present / absent
    'search stimuli (classify)',
    'ImageNet',
    'Stylized ImageNet',
    'DomainNet, Clipart domain',
)


def main(results_dir,
         results_csv_filename
         ):
    results_dir = Path(results_dir)
    results_csv_path = results_dir / results_csv_filename
    df = pd.read_csv(results_csv_path)

    df = df[df.target_condition == 'both']  # note that we keep only 'both' condition!
    # do *not* keep random weights, and training just on `searchstims` themselves
    df = df[df.source_dataset.isin(SOURCE_DATASETS_TO_USE)]

    # factorize method + source dataset, to iterate over in slightly more concise way
    df['method_source_dataset'] = pd.factorize(pd._lib.fast_zip(
        [df.method.values, df.source_dataset.values]
    ))[0]

    # need to compute 'discrim_pct' differently for the two different stimulus types
    df_rvg = df[df['stimulus'].str.contains('RVvGV')].copy()
    df_rvg['greenness'] = df_rvg['stimulus'].apply(lambda x: int(x.split('_')[-1]))
    df_rvg['discrim_pct'] = 1 - (df_rvg.greenness / 255.)

    df_tvt = df[df['stimulus'].str.contains('TvT')].copy()
    df_tvt['rotation'] = df_tvt['stimulus'].apply(lambda x: int(x.split('_')[-1]))
    df_tvt['discrim_pct'] = df_tvt.rotation / 90.

    for stim_df, stim_type in zip(
            (df_rvg, df_tvt),
            ('rvg', 'tvt')
    ):
        # ## Fit a psychometric function
        fits_df = fits_df_from_discrim_df(stim_df)

        # ## Use the fit parameters to find the desired difference threshold,
        # i.e. what target-distractor discriminability gives us a difference threshold of X?
        # Here the threshold was 75%, following Palmer et al. 2000 and references therein
        # (https://www.sciencedirect.com/science/article/pii/S0042698999002448)

        ## Log transform, then find mean difference threshold across 'subjects' (training replicates)
        # so we can do linear regression on set size v. difference threshold
        fits_df['log_set_size'] = np.log10(fits_df.set_size)
        fits_df['log_discrim_threshold'] = np.log10(fits_df.discrim_threshold)

        # ### find slopes
        slopes_df = regress_net_number(fits_df)
        slopes_df = slopes_df.dropna()
        slopes_gb = slopes_df.groupby(by=['net_name', 'source_dataset']).agg(
            slope=pd.NamedAgg('slope', aggfunc='mean'),
            intercept=pd.NamedAgg('intercept', aggfunc='mean')
        )
        slopes_agg_df = slopes_gb.reset_index()
        slopes_agg_df = slopes_agg_df.sort_values(by=['net_name', 'slope'])

        stim_df.to_csv(
            results_dir / f'source_fits_{stim_type}.csv'
        )
        fits_df.to_csv(
            results_dir / f'fits_{stim_type}.csv'
        )
        slopes_df.to_csv(
            results_dir / f'slopes_{stim_type}.csv'
        )
        slopes_agg_df.to_csv(
            results_dir / f'slopes_agg_{stim_type}.csv'
        )


RESULTS_DIR = pyprojroot.here() / 'results' / 'searchstims' / 'source_data' / 'discriminability'
SOURCE_RESULTS_CSV_FILENAME = 'all_discrim_expts.csv'


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--results_dir', default=RESULTS_DIR
    )
    parser.add_argument(
        '--source_results_csv_filename', default=SOURCE_RESULTS_CSV_FILENAME
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(results_dir=args.results_dir,
         results_csv_filename=args.source_results_csv_filename)
