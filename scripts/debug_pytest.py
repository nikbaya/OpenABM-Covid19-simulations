#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 10:15:48 2021

Debug pytest failures

@author: nbaya
"""

import os, subprocess, shutil
import numpy as np, pandas as pd
from COVID19.parameters import ParameterSet


OPENABM_DIR = '/Users/nbaya/gms/fraser_lab/OpenABM-Covid19'

SUSCEPTIBLE = 0

os.chdir(OPENABM_DIR)
from tests import constant
from tests import utilities as utils
from tests import test_infection_dynamics
from tests import test_network

def setup_covid_methods():
    """
    Called before each method is run; creates a new data dir, copies test datasets
    """
    os.mkdir(constant.DATA_DIR_TEST)
    shutil.copy(constant.TEST_DATA_TEMPLATE, constant.TEST_DATA_FILE)
    shutil.copy(constant.TEST_HOUSEHOLD_TEMPLATE, constant.TEST_HOUSEHOLD_FILE)
    shutil.copy(constant.TEST_HOSPITAL_TEMPLATE, constant.TEST_HOSPITAL_FILE)
    
    # Adjust any parameters that need adjusting for all tests
    params = ParameterSet(constant.TEST_DATA_FILE, line_number=1)
    params.set_param("n_total", 10000)
    params.set_param("end_time", 1)
    params.write_params(constant.TEST_DATA_FILE)

def compile_covid_ibm():
    """
    Compile the IBM in a temporary directory
    """
    # Make a temporary copy of the code 
    # (remove this temporary directory if it already exists)
    shutil.rmtree(constant.IBM_DIR_TEST, ignore_errors = True)
    shutil.copytree(constant.IBM_DIR, constant.IBM_DIR_TEST)
    
    # Construct the compilation command and compile
    compile_command = "make clean; make install"
    completed_compilation = subprocess.run(
        [compile_command], shell = True, cwd = constant.IBM_DIR_TEST, capture_output = True
    )

def set_up():
    setup_covid_methods()
    compile_covid_ibm()

def destroy():
    shutil.rmtree(constant.DATA_DIR_TEST, ignore_errors=True)
    shutil.rmtree(constant.IBM_DIR_TEST, ignore_errors = True)

def test_monoton_mild_infectious_factor(
            self,
            **kwargs,    
        ):
        """
        Test that monotonic change (increase, decrease, or equal) in mild_infectious_factor values
        leads to corresponding change (increase, decrease, or equal) in the total infections.
        
        """
        
        def mean_total_infected(params, mild_infectious_factor, rng_seed_range=range(1,21)):
            """
            Run simulation with parameters `params`, mild_infectious_factor=`mild_infectious_factor`,
            and for all rng_seed values in `rng_seed_range`. 
            Returns mean of total_infected from final day of simulation, across 
            all seeds in `rng_seed_range`.
            """
            params.set_param('mild_infectious_factor', mild_infectious_factor)
            total_infected_list = []
            for rng_seed in rng_seed_range:
                params.set_param('rng_seed', rng_seed)
                params.write_params(constant.TEST_DATA_FILE)     
                
                file_output   = open(constant.TEST_OUTPUT_FILE, "w")
                completed_run = subprocess.run([constant.command], stdout = file_output, shell = True)     
                df_output     = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
            
                total_infected_list.append(df_output[ "total_infected" ].iloc[-1])
            
            return np.mean(total_infected_list)
        
        # calculate the total infections for the first entry in the asymptomatic_infectious_factor values
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        mild_infectious_factor_current = mean_total_infected(params, mild_infectious_factor[0])
        
        print('current\tnew\t\tcurrent\tnew')        
        # calculate the total infections for the rest and compare with the current
            
        
        for idx in range(1, len(kwargs['mild_infectious_factor'])):
            for k,v in kwargs.items():
                if k=="mild_infectious_factor":
                    params.set_param(k, v[idx])
                else:
                    params.set_param(k, v)
                
            mild_infectious_factor_new = kwargs['mild_infectious_factor'][idx]
            total_infected_list_new = []
            
            for rng_seed in rng_seed_range:
                params.set_param('rng_seed', rng_seed)
                params.write_params(constant.TEST_DATA_FILE)
                
                file_output   = open(constant.TEST_OUTPUT_FILE, "w")
                _             = subprocess.run([constant.command], stdout = file_output, shell = True)
                df_output_new = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
                
                total_infected_new = df_output_new[ "total_infected" ].iloc[-1]
                total_infected_list_new.append(total_infected_new)
            
            mean_total_infected_new     = np.mean(total_infected_list_new)
            min_current, max_current = min(total_infected_list_current), max(total_infected_list_current)
            min_new, max_new = min(total_infected_list_new), max(total_infected_list_new)
            print(f'{mild_infectious_factor_current}\t\t{mild_infectious_factor_new}'+
                  f'\t\t{mean_total_infected_current}\t\t{mean_total_infected_new}\t'+
                  f'({min_current}-{max_current})\t({min_new}-{max_new})')
            # check the total infections
            # if mild_infectious_factor_new > mild_infectious_factor_current:
            #     np.testing.assert_equal( total_infected_new > total_infected_current, True)
            # elif mild_infectious_factor_new < mild_infectious_factor_current:
            #     np.testing.assert_equal( total_infected_new < total_infected_current, True)
            # elif mild_infectious_factor_new == mild_infectious_factor_current:
            #     np.testing.assert_allclose( total_infected_new, total_infected_current, atol = 0.01)
            
            # check the total infections
            if mild_infectious_factor_new > mild_infectious_factor_current:
                np.testing.assert_equal( mean_total_infected_new > mean_total_infected_current, True)
            elif mild_infectious_factor_new < mild_infectious_factor_current:
                np.testing.assert_equal( mean_total_infected_new < mean_total_infected_current, True)
            elif mild_infectious_factor_new == mild_infectious_factor_current:
                np.testing.assert_allclose( mean_total_infected_new, mean_total_infected_current, atol = 0.01)
            
            # refresh current values
            mild_infectious_factor_current = mild_infectious_factor_new
            mean_total_infected_current = mean_total_infected_new

def test_infectiousness_multiplier( self, kwargs, rng_seed_range=range(1,2)):
    """
    Check that the total infected stays the same up to 0.5 SD.
    """
    test_params = kwargs['test_params']
    sd_multipliers = kwargs['sd_multipliers']
    
    ordered_multipliers = sorted( sd_multipliers )
    # transmissions = []
    total_infected_means = []
    total_infected_range = []
    for idx, sd_multiplier in enumerate(ordered_multipliers):
        total_infected = []
        for rng_seed in rng_seed_range:
            params = utils.get_params_swig()
            for param, value in test_params.items():
                params.set_param( param, value )  
            params.set_param( "sd_infectiousness_multiplier", sd_multiplier )
            params.set_param( "rng_seed", rng_seed )
            model  = utils.get_model_swig( params )
            # print(f'rng_seed: {model.c_params.rng_seed}')
              
            for time in range( test_params[ "end_time" ] ):
                model.one_time_step()
              
            results = model.one_time_step_results()
            total_infected.append( results[ "total_infected" ] )
              
            del model
            del params

        total_infected_means.append(np.mean(total_infected))        
        total_infected_range.append(f'{min(total_infected)}-{max(total_infected)}')
        print(str(total_infected_means[0])+'\t'+str(total_infected_means[idx])+
              '\t'+str(abs(total_infected_means[0]-total_infected_means[idx])/total_infected_means[idx])+
              '\t'+total_infected_range[idx])

    # np.testing.assert_allclose([base_infected]*len(total_infected), total_infected, rtol=0.05)
        
    base_infected_mean = total_infected_means[0]
    print(rng_seed_range)
    print('\t'.join(([str(base_infected_mean)]*len(total_infected_means))))
    print('\t'.join([str(x) for x in total_infected_means]))
    print('\t'.join(['{0:.4f}'.format(abs(x-y)/y) for x,y in zip([base_infected_mean]*len(total_infected_means), total_infected_means)]))

def test_custom_occupation_network( self, test_params, rng_seed_range=range(1,2) ):
    """
      For user defined occupational networks,
      check to see that people only meet with the same person
      once per day on each occupational network;

      Check to see that when you look over multiple days that
      the mean number of unique contacts is mean_daily/daily_fraction
    """

    # Set up user-defined occupation network tables.
    n_total = test_params['n_total']
    IDs = np.arange(n_total, dtype='int32')
    network_no = np.arange(10, dtype='int32')  # Set up 10 occupation networks

    assignment = np.zeros(n_total, dtype='int32')
    for i in range(10):
        assignment[i*n_total//10 : (i+1)*n_total//10] = i

    age_type = np.zeros(10)
    age_type[0:2] = constant.CHILD
    age_type[2:8] = constant.ADULT
    age_type[8:10] = constant.ELDERLY

    mean_work_interaction = np.zeros(10)
    mean_work_interaction[0:2] = test_params['mean_work_interactions_child']
    mean_work_interaction[2:8] = test_params['mean_work_interactions_adult']
    mean_work_interaction[8:] = test_params['mean_work_interactions_elderly']

    lockdown_multiplier = np.ones(10) * 0.2

    network_id = np.arange(10)
    network_name = ['primary', 'secondary', 'adult_1', 'adult_2', 'adult_3', 'adult_4',
                    'adult_5', 'adult_6', 'elderly_1', 'elderly_2']

    df_occupation_network  = pd.DataFrame({'ID':IDs,'network_no':assignment})
    df_occupation_network_property = pd.DataFrame({
        'network_no': network_no,
        'age_type': age_type,
        'mean_work_interaction': mean_work_interaction,
        'lockdown_multiplier': lockdown_multiplier,
        'network_id': network_id,
        'network_name': network_name})

    params = utils.get_params_swig()
    for param, value in test_params.items():
        params.set_param( param, value )
    # load custom occupation network table before constructing the model
    params.set_occupation_network_table(df_occupation_network, df_occupation_network_property)

    # make a simple demographic table. For small networks, household rejection sampling won't converge.
    hhIDs      = np.array( range(n_total), dtype='int32')
    house_no = np.array( hhIDs / 4, dtype='int32' )
    ages     = np.array( np.mod( hhIDs, 9) , dtype='int32' )
    df_demo  = pd.DataFrame({'ID': hhIDs,'age_group':ages,'house_no':house_no})

    # add to the parameters and get the model
    params.set_demographic_household_table( df_demo ),

    model  = utils.get_model_swig( params )
    model.one_time_step()
    model.write_interactions_file()
    df_inter = pd.read_csv(constant.TEST_INTERACTION_FILE)
    df_inter[ "time" ] = 0

    for time in range( test_params[ "end_time" ] ):
        model.one_time_step()
        model.write_interactions_file()
        df = pd.read_csv(constant.TEST_INTERACTION_FILE)
        df[ "time" ] = time + 1
        df_inter = df_inter.append( df )

    df_inter = df_inter[ df_inter[ "type" ] == constant.OCCUPATION ]

    # check to see there are sufficient daily connections and only one per set of contacts a day
    df_unique_daily = df_inter.groupby( ["time","ID_1","ID_2"]).size().reset_index(name="N")

    connection_upper_bound = (test_params["n_total"] / 10 -1 ) // 2 * 2.0

    min_size = (test_params["end_time"]+1) * test_params["n_total"] * (
            0.2 * min (connection_upper_bound * test_params['daily_fraction_work'], test_params["mean_work_interactions_child"] )
            + 0.6 * min (connection_upper_bound * test_params['daily_fraction_work'], test_params["mean_work_interactions_adult"])
            + 0.2 * min (connection_upper_bound * test_params['daily_fraction_work'], test_params["mean_work_interactions_elderly"]))

    # np.testing.assert_allclose(sum(df_unique_daily["N"]==1), min_size, rtol=0.1, err_msg="Unexpected contacts on the occupational networks" )
    # np.testing.assert_equal(sum(df_unique_daily["N"]!=1), 0, "Repeat connections on same day on the occupational networks" )
    print(sum(df_unique_daily["N"]==1), min_size, '{0:.5f}'.format(abs(sum(df_unique_daily["N"]==1)-min_size)/min_size))
    rtol = 0.1 if n_total>30 else 0.15
    try:
        np.testing.assert_allclose(sum(df_unique_daily["N"]==1), min_size, rtol=rtol, err_msg="Unexpected contacts on the occupational networks" )
    except:
        print(f'Failed: Unexpected contacts on the occupational networks (rng_seed={test_params["rng_seed"]})')
    np.testing.assert_equal(sum(df_unique_daily["N"]!=1), 0, "Repeat connections on same day on the occupational networks" )

    # check the mean unique connections over multiple days is mean/daily fraction
    df_unique = df_inter.groupby(["occupation_network_1","ID_1","ID_2"]).size().reset_index(name="N_unique")
    df_unique = df_unique.groupby(["occupation_network_1","ID_1"]).size().reset_index(name="N_conn")
    df_unique = df_unique.groupby(["occupation_network_1"]).mean()

    mean_by_type = [ test_params["mean_work_interactions_child"],test_params["mean_work_interactions_adult"],test_params["mean_work_interactions_elderly"]]

    for network in range(10): # 10 custom occupation networks
        actual   = df_unique.loc[network,{"N_conn"}]["N_conn"]
        expected = min( connection_upper_bound,
                        mean_by_type[constant.CUSTOM_NETWORK_TYPE_MAP[network]]/test_params["daily_fraction_work"])
        if expected == connection_upper_bound:
            atol = 1
            np.testing.assert_allclose(actual,expected,atol=atol,err_msg="Expected mean unique occupational contacts over multiple days not as expected")
        else:
            rtol = 0.02
            np.testing.assert_allclose(actual,expected,rtol=rtol,err_msg="Expected mean unique occupational contacts over multiple days not as expected")


def main():
    destroy()
    set_up()
    kwargs = test_infection_dynamics.TestClass.params['test_monoton_mild_infectious_factor'][0].copy()
    test_monoton_mild_infectious_factor(self=test_infection_dynamics.TestClass, kwargs=kwargs, rng_seed_range=range(1,21))
    # kwargs = test_infection_dynamics.TestClass.params['test_infectiousness_multiplier'][0].copy()
    # test_infectiousness_multiplier(self=test_infection_dynamics.TestClass, kwargs=kwargs, rng_seed_range=range(1,21))
    
    kwargs = test_network.TestClass.params['test_custom_occupation_network'].copy()
    test_params_list = [x['test_params'] for x in kwargs]
    for rng_seed in range(1,21):
        print(f'rng_seed: {rng_seed}')
        # for idx, test_params in enumerate(test_params_list, 1):
        test_params = test_params_list[-1]
        test_params['rng_seed'] = rng_seed
        test_custom_occupation_network(self=test_network.TestClass, test_params=test_params)        
        
    destroy()
