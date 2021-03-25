#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:42:32 2021

Run simple multi-strain simulation

@author: nbaya
"""

from COVID19.model import Model, Parameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OPENABM_DIR = '/Users/nbaya/gms/fraser_lab/OpenABM-Covid19'
DATA_DIR_TEST = '/Users/nbaya/Downloads/tmp_openabm'
PLOT_DIR = '/Users/nbaya/Downloads'

TEST_DATA_TEMPLATE = f"{OPENABM_DIR}/tests/data/baseline_parameters.csv"
TEST_HOUSEHOLD_TEMPLATE = f"{OPENABM_DIR}/tests/data/baseline_household_demographics.csv"
TEST_HOSPITAL_FILE = f"{OPENABM_DIR}/tests/data/hospital_baseline_parameters.csv"

TEST_INDIVIDUAL_FILE = f"{DATA_DIR_TEST}/individual_file_Run1.csv"
TEST_TRANSMISSION_FILE = f"{DATA_DIR_TEST}/transmission_Run1.csv"

SUSCEPTIBLE = 0

def get_params_swig():
    return Parameters(
        TEST_DATA_TEMPLATE,
        1,
        DATA_DIR_TEST,
        TEST_HOUSEHOLD_TEMPLATE,
        TEST_HOSPITAL_FILE,
        1,
        True,
        True
    )

def get_model_swig( params ):
    return Model( params )

def test_multiple_strain_domination(test_params, n_extra_infections, t_extra_infections, t_check_after, strain_multiplier ):
    # copied from test suite: https://github.com/nikbaya/OpenABM-Covid19/blob/97574a285840a977797d184c0dffc86f33e327ce/tests/test_infection_dynamics.py
    params = get_params_swig()
    for param, value in test_params.items():
        params.set_param( param, value )  
    
    model  = get_model_swig( params )
    
    for time in range(t_extra_infections):
        model.one_time_step()
    
    # randomly infect susceptible individual
    model.write_individual_file()
    df_indiv = pd.read_csv( TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
    idxs     = df_indiv[ df_indiv[ "current_status" ] == SUSCEPTIBLE ]['ID'].to_numpy()
    n_susc   = len( idxs )
    
    inf_id = np.random.choice( n_susc, n_extra_infections, replace=False)
    for idx in range( n_extra_infections ):
        model.seed_infect_by_idx( idxs[ inf_id[ idx ] ], strain_multiplier = strain_multiplier )
    
    for time in range(test_params["end_time"] - t_extra_infections):
        model.one_time_step()
    
    # get the new infections for each time step for each strain    
    model.write_transmissions()
    df_trans = pd.read_csv( TEST_TRANSMISSION_FILE, comment="#", sep=",", skipinitialspace=True )  
    df_n_trans = df_trans.loc[:,["time_infected","strain_multiplier"]]
    df_n_trans = df_n_trans.pivot_table( index = ['time_infected'], columns = ["strain_multiplier"], aggfunc=len).fillna(0).reset_index() 
    
    # # check no new strain infections before it is introduced
    # np.testing.assert_equal( df_n_trans[ df_n_trans["time_infected"] < t_extra_infections ][ strain_multiplier ].sum(), 0, "new strain cases before seed date" )
    
    # # check that the new strain dominates after a set period of time
    # n_base = df_n_trans[ df_n_trans["time_infected"] > t_extra_infections + t_check_after][ 1.0 ].sum()
    # n_new  = df_n_trans[ df_n_trans["time_infected"] > t_extra_infections + t_check_after][ strain_multiplier ].sum()
    # np.testing.assert_array_less( 0.90, n_new / ( n_new + n_base), "new strain is less than 90% of new cases")
    
    return df_n_trans

def main():
    test_params = dict(rng_seed = 1,
                       n_total = 1e4,
                       n_seed_infection = 5,
                       end_time = 80,
                       infectious_rate = 3
                    )
    n_extra_infections = 5
    t_extra_infections = 20
    t_check_after      = 30 # time after the new strain to check for domination of second strain
    strain_multiplier  = 2.0
    
    df_n_trans= test_multiple_strain_domination(
            test_params=test_params, 
            n_extra_infections=n_extra_infections, 
            t_extra_infections=t_extra_infections, 
            t_check_after=t_check_after, 
            strain_multiplier=strain_multiplier
    )
    
    plt.figure()
    min_y = 0
    for multiplier in [1, 2]:
        trans_per_strain = df_n_trans[:][multiplier]
        plt.bar(df_n_trans.time_infected, trans_per_strain, bottom=min_y, label = f'strain_multiplier={multiplier}')
        min_y = trans_per_strain
    
    ylim = plt.ylim()
    plt.plot([t_extra_infections]*2, ylim, 'k--', label = 'extra infections introduced')
    plt.xlabel('days')
    plt.ylabel('transmissions')
    plt.legend(loc='upper left')
    plt.title(f'seed: {test_params["rng_seed"]}')
    plt.savefig(f'{PLOT_DIR}/test_multiple_strains.seed_{test_params["rng_seed"]}.png', dpi=300)
    
if __name__=='__main__':
    main()
