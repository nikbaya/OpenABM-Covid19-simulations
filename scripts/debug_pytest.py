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
            kwargs,    
        ):
        """
        Test that monotonic change (increase, decrease, or equal) in mild_infectious_factor values
        leads to corresponding change (increase, decrease, or equal) in the total infections.
        
        """
        
        # calculate the total infections for the first entry in the asymptomatic_infectious_factor values
        params = ParameterSet(constant.TEST_DATA_FILE, line_number = 1)
        for k,v in kwargs.items():
            if k=="mild_infectious_factor":
                params.set_param(k, v[0])
            else:
                params.set_param(k, v)

        mild_infectious_factor_current = kwargs['mild_infectious_factor'][0]
        total_infected_list_current = []
        
        for rng_seed in range(1,11):
            params.set_param('rng_seed', rng_seed)
            params.write_params(constant.TEST_DATA_FILE)     
            
            file_output   = open(constant.TEST_OUTPUT_FILE, "w")
            _             = subprocess.run([constant.command], stdout = file_output, shell = True)     
            df_output     = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
            
            # save the current mild_infectious_factor value
            total_infected_current = df_output[ "total_infected" ].iloc[-1]
            total_infected_list_current.append(total_infected_current)
        
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
            
            for rng_seed in range(1,11):
                params.set_param('rng_seed', rng_seed)
                params.write_params(constant.TEST_DATA_FILE)
                
                file_output   = open(constant.TEST_OUTPUT_FILE, "w")
                _             = subprocess.run([constant.command], stdout = file_output, shell = True)
                df_output_new = pd.read_csv(constant.TEST_OUTPUT_FILE, comment = "#", sep = ",")
                
                total_infected_new = df_output_new[ "total_infected" ].iloc[-1]
                total_infected_list_new.append(total_infected_new)
            
            mean_current = np.mean(total_infected_list_current)
            mean_new     = np.mean(total_infected_list_new)
            min_current, max_current = min(total_infected_list_current), max(total_infected_list_current)
            min_new, max_new = min(total_infected_list_new), max(total_infected_list_new)
            print(f'{mild_infectious_factor_current}\t\t{mild_infectious_factor_new}\t\t{mean_current}\t\t{mean_new}\t({min_current}-{max_current})\t({min_new}-{max_new})')
            # check the total infections
            # if mild_infectious_factor_new > mild_infectious_factor_current:
            #     np.testing.assert_equal( total_infected_new > total_infected_current, True)
            # elif mild_infectious_factor_new < mild_infectious_factor_current:
            #     np.testing.assert_equal( total_infected_new < total_infected_current, True)
            # elif mild_infectious_factor_new == mild_infectious_factor_current:
            #     np.testing.assert_allclose( total_infected_new, total_infected_current, atol = 0.01)
            
            # refresh current values
            mild_infectious_factor_current = mild_infectious_factor_new
            total_infected_list_current = total_infected_list_new

def test_infectiousness_multiplier( self, kwargs, rng_seed_range=range(1,11)):
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
        
    base_infected_mean = total_infected_means[0]
    print(rng_seed_range)
    print('\t'.join(([str(base_infected_mean)]*len(total_infected_means))))
    print('\t'.join([str(x) for x in total_infected_means]))
    print('\t'.join(['{0:.4f}'.format(abs(x-y)/y) for x,y in zip([base_infected_mean]*len(total_infected_means), total_infected_means)]))

    # np.testing.assert_allclose([base_infected]*len(total_infected), total_infected, rtol=0.05)


def main():
    destroy()
    set_up()
    # kwargs = test_infection_dynamics.TestClass.params['test_monoton_mild_infectious_factor'][0].copy()
    # test_monoton_mild_infectious_factor(self=test_infection_dynamics.TestClass, kwargs=kwargs)
    kwargs = test_infection_dynamics.TestClass.params['test_infectiousness_multiplier'][0].copy()
    test_infectiousness_multiplier(self=test_infection_dynamics.TestClass, kwargs=kwargs, rng_seed_range=range(1,3))
    destroy()
