#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:25:23 2021

For plotting OpenABM-Covid19 results

@author: nbaya
"""


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import scipy.stats as stats
import argparse

WD = '/users/gms/whv244/fraser_lab/OpenABM-Covid19-simulations'
# DATA_DIR = '/Users/nbaya/Downloads'
# PLOTS_DIR = '/Users/nbaya/Downloads'
DATA_DIR = f'{WD}/data'
PLOTS_DIR = f'{WD}/plots'

def get_sim_results_path(prefix, seed, strain_info=False):
    return f'{DATA_DIR}/{prefix}{".strain_info" if strain_info else ""}.seed_{seed}.{"tsv" if strain_info else "csv"}'

def read_all_results(prefix, seeds=range(1,21), include_strain_info=True):
    r'''Return list of results tables from multiple replicates with different seeds
    '''
    read_csv = lambda seed: pd.read_csv(get_sim_results_path(prefix, seed), comment="#").dropna(axis=0).astype(int)
    df_list = [read_csv(seed) for seed in seeds]
    _ = [d.insert(0,'seed',seed) for d, seed in zip(df_list, seeds)]
    if include_strain_info:
        try:
            read_strain_info = lambda seed: pd.read_csv(get_sim_results_path(prefix, seed, strain_info=True), sep='\t')
            strain_info_df_list = [read_strain_info(seed) for seed in seeds]
            df_list = [pd.concat((df, strain_info_df), axis=1) for df, strain_info_df in zip(df_list, strain_info_df_list)]
        except FileNotFoundError:
            print(f'Strain info files missing for prefix={prefix}')
    return df_list

def plot_days_vs_field(df_list, field, prefix, ax=None, plot_CI=True, save=False, base_color='k'):
    r'''Plot a field as a function of days for every replicate with a different seed
    
    If plot_CI=True, plot 95% confidence interval for mean, along with the 
    mean across replicates.
    '''
    n_reps = len(df_list) # number of replicates with different seeds for the random number generator
    x = df_list[0]['time'] # assume all sims ran for the same amount of time
    y = pd.concat([df[field] for df in df_list], axis=1)
    if ax is None:
        fig,ax = plt.subplots(figsize=(6,4))
    if plot_CI:
        y_mean = y.mean(axis=1)
        y_se = stats.sem(y, axis=1)
        ax.fill_between(x, y_mean-1.96*y_se, y_mean+1.96*y_se, color=base_color, 
                        lw=0, alpha=0.2, label='95% CI for mean')
        ax.plot(x, y_mean, lw=1, c=base_color, label='mean')
        plt.legend()
    else:
        ax.plot(x, y,'-')
    ax.set_ylabel(field)
    ax.set_xlabel('days')
    ax.set_title(f'{field}\n{prefix}, reps={n_reps}')
    plt.tight_layout()
    if save:
        plt.savefig(f'{PLOTS_DIR}/{prefix}.{field}.reps_{n_reps}{".plotCI" if plot_CI else ""}.png', dpi=300)
        
def plot_selected_fields(df_list, prefix, fields, plot_CI=True, save=False):
    r'''Make subplots of multiple fields across time
    '''
    pass

def plot_sim_comparison(prefix_list, field, plot_CI=True, save=False, alpha=1):
    r'''Plot comparison of the same field between two simulations
    '''
    df_list_list = [read_all_results(p) for p in prefix_list]
    n_reps = len(df_list_list[0]) # number of replicates with different seeds for the random number generator
    x = df_list_list[0][0]['time'] # assume all sims ran for the same amount of time
    y_list = [pd.concat([df[field] for df in df_list], axis=1) for df_list in df_list_list]
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(prefix_list)]

    fig, ax = plt.subplots(figsize=(6,4))
    for y, prefix, color in zip(y_list, prefix_list, color_list):
        if plot_CI:
            y_mean = y.mean(axis=1)
            y_se = stats.sem(y, axis=1)
            ax.fill_between(x, y_mean-1.96*y_se, y_mean+1.96*y_se, color=color, 
                            lw=0, alpha=0.2)
            ax.plot(x, y_mean, lw=1, c=color, label=f'{prefix}')
            plt.legend()
        else:
            ax.plot(x, y,'-', color=color, alpha=alpha, lw=0.5)
    if not plot_CI:
        custom_lines = [Line2D([0], [0], color=c, lw=2) for c in color_list]
        plt.legend(custom_lines, prefix_list)
    ax.set_ylabel(field)
    ax.set_xlabel('days')
    ax.set_title(f'{field}\nreps={n_reps}')
    plt.tight_layout()
    if save:
        plt.savefig(f'{PLOTS_DIR}/{"_vs_".join(prefix_list)}.{field}.reps_{n_reps}{".plotCI" if plot_CI else ""}.png', dpi=300)

def plot_sim_comparison_2fields(prefix_list, fields, save=False):
    r'''Plot comparison of two fields between two simulations
    '''
    assert len(fields)==2, '`fields` must be a list of only two fields'
    df_list_list = [read_all_results(p) for p in prefix_list]
    n_reps = len(df_list_list[0]) # number of replicates with different seeds for the random number generator
    x_list = [pd.concat([df[fields[0]] for df in df_list], axis=1) for df_list in df_list_list]
    y_list = [pd.concat([df[fields[1]] for df in df_list], axis=1) for df_list in df_list_list]
    color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(prefix_list)]

    fig, ax = plt.subplots(figsize=(6,4))
    for x, y, prefix, color in zip(x_list, y_list, prefix_list, color_list):
        ax.plot(x, y,'-', color=color, alpha=1, lw=0.5)
    custom_lines = [Line2D([0], [0], color=c, lw=2) for c in color_list]
    plt.legend(custom_lines, prefix_list)
    ax.set_xlabel(fields[0])
    ax.set_ylabel(fields[1])
    ax.set_title(f'{fields[0]} vs. {fields[1]}\nreps={n_reps}')
    plt.tight_layout()
    if save:
        plt.savefig(f'{PLOTS_DIR}/{"_vs_".join(prefix_list)}.{"_vs_".join(fields)}.reps_{n_reps}.png', dpi=300)

def plot_strain_bins(df, proportion=True, save=False):
    r'''Plot proportion of strains that fall in bins of `transmission_multiplier`
    '''
    n_days, n_bins = df.shape
    x = range(1, n_days+1)
    cmap = cm.get_cmap('coolwarm', lut=n_bins)
    max_bin_lower_lim = 2
    labels = [f'[{i/((n_bins-1)/max_bin_lower_lim)}, {(i+1)/((n_bins-1)/max_bin_lower_lim)})' for i in range(n_bins)]
    vals = df.values
    if proportion:
        row_sums = vals.sum(axis=1)
        vals = vals/row_sums[:,None]
    y = 0
    min_y = 0
    plt.figure(figsize=(12,8))
    for bin_idx in range(n_bins):
        y = vals[:,bin_idx]
        if all(y==0):
            continue
        plt.bar(x, y, width=1, bottom=min_y, color=cmap(bin_idx), label=labels[bin_idx])
        # plt.fill_between(x, min_y, min_y+y, color=cmap(bin_idx), label=labels[bin_idx])
        min_y += y
    plt.legend(loc='upper left', title='transmission_multiplier', title_fontsize='small')
    plt.xlabel('days')
    plt.ylabel(f'{"proportion" if proportion else "count"} of infections caused')
    plt.title('Infections caused by strains binned by transmission_multiplier')
    plt.xlim(min(x), max(x))
    if proportion:
        plt.ylim(0,1)
    plt.tight_layout()
    if save:
        plt.savefig(f'{PLOTS_DIR}/strain_bins.v3{".proportion" if proportion else ""}.png', dpi=300)

def main(args):
    prefix = args.prefix
    field = args.plot_field
    compare_sims = args.compare_sims
    prefix2 = args.prefix2
    
    df_list = read_all_results(prefix)
    
    if field is not None:
        plot_days_vs_field(df_list, field, prefix, save=True)
        
    if compare_sims:
        prefix1 = prefix
        plot_sim_comparison(prefix1, prefix2, field, plot_CI=True, save=True)
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default=None, help='Prefix of sim results to load (e.g. "default_sim")')
    parser.add_argument('--plot_field', default=None, help='Make a plot for the specified field (e.g. "total_infected")')
    parser.add_argument('--plot_all_fields', action='store_true', help='Make a plot for a single field')
    parser.add_argument('--compare_sims', action='store_true', help='Make a plot comparing two scenarios')
    parser.add_argument('--prefix2', default=None, help='Prefix of second sim results to load (e.g. "default_sim_transmul1.5")')
    args = parser.parse_args()

    main(args)

prefix1 = 'default_sim'
    

prefix_list = ['default_sim',
               'default_sim_transmul1.1',
               'default_sim_transmul1.5']
plot_sim_comparison(prefix_list, field='total_case', plot_CI=False, save=True)
plot_sim_comparison(prefix_list, field='n_symptoms', plot_CI=False, save=True)

prefix_list = ['default_sim', 
               'default_5seedstrains_transmul1']
plot_sim_comparison(prefix_list, field='total_case', plot_CI=False, save=True)
plot_sim_comparison(prefix_list, field='n_symptoms', plot_CI=False, save=True)

prefix_list = ['default_1seed_p0.0001_N1_0.001',
               'default_1seed_p0.0001_N1_0.01',
               'default_1seed_p0.0001_N1_0.1']
plot_sim_comparison(prefix_list, field='total_case', plot_CI=False, save=True)
plot_sim_comparison(prefix_list, field='n_symptoms', plot_CI=False, save=True, alpha=0.5)

prefix_list = ['default_1seed_p0.0001_N1_0.1',
               'default_1seed_p0.001_N1_0.1']
plot_sim_comparison(prefix_list, field='mean_transmission_multiplier', plot_CI=False, save=True)
plot_sim_comparison(prefix_list, field='total_strains', plot_CI=False, save=True)
plot_sim_comparison(prefix_list, field='total_case', plot_CI=False, save=True)
plot_sim_comparison_2fields(prefix_list, fields=['total_strains','total_case'], save=True)


prefix2 = 'default_1seed_p0.0001_N1_0.01'
prefix2 = 'default_1seed_p0.0001_N1_0.001'

prefix_list = ['default_1seed_p0.0001_Nparent_0.1',
               'default_1seed_p0.001_Nparent_0.1']
plot_sim_comparison(prefix_list, field='mean_transmission_multiplier', plot_CI=False, save=True)
plot_sim_comparison(prefix_list, field='total_strains', plot_CI=False, save=True)
plot_sim_comparison(prefix_list, field='total_case', plot_CI=False, save=True)
plot_sim_comparison_2fields(prefix_list, fields=['total_strains','total_case'], save=True)


prefix1 = 'default_1seed_p0.0001_N1_0.1'
prefix2 = 'default_1seed_p0.0001_N1.1_0.1'


# df = pd.read_csv(f'{DATA_DIR}/tmp_openabm/openabm_50seedinfection_infectrate2_mutprob0.001_Nparent_0.1.bins_w_strain_infections.tsv',
#                  delim_whitespace=True, names=[f'bin_{i}' for i in range(21)])

df = pd.read_csv(f'{DATA_DIR}/tmp_openabm/openabm_50seedinfection_infectrate2_mutprob0.001_Nparent_0.1.bins_w_strain_infections.updated.tsv',
                 delim_whitespace=True, names=[f'bin_{i}' for i in range(21)])

# df = pd.read_csv(f'{DATA_DIR}/tmp_openabm/openabm_50seedinfection_infectrate2_mutprob0.001_Nparent_0.1.bins_w_strains.tsv',
#                  delim_whitespace=True, names=[f'bin_{i}' for i in range(21)])

df = pd.read_csv(f'{DATA_DIR}/tmp_openabm//openabm_50seedinfection_infectrate1.8_mutprob0.001_Nparent_0.1.490days.tsv',
                 delim_whitespace=True, names=[f'bin_{i}' for i in range(21)])
df = pd.read_csv(f'{DATA_DIR}/tmp_openabm//openabm_50seedinfection_infectrate1.6_mutprob0.001_Nparent_0.1.490days.tsv',
                 delim_whitespace=True, names=[f'bin_{i}' for i in range(21)])
plot_strain_bins(df, proportion=True, save=True)
plot_strain_bins(df, proportion=False, save=True)

df = pd.read_csv(f'{DATA_DIR}/tmp_openabm//openabm_50seedinfection_infectrate1.7_mutprob0.001_Nparent_0.1.490days.tsv',
                 delim_whitespace=True, names=[f'bin_{i}' for i in range(21)])
plot_strain_bins(df, proportion=False, save=True)
plot_strain_bins(df, proportion=True, save=True)

df = pd.read_csv(f'{DATA_DIR}/tmp_openabm/openabm_50seedinfection_infectrate1_mutprob0.001_Nparent_0.5.490days.tsv',
                 delim_whitespace=True, names=[f'bin_{i}' for i in range(21)])
plot_strain_bins(df, proportion=False, save=True)
plot_strain_bins(df, proportion=True, save=True)


