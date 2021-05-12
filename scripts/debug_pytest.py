#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 10:15:48 2021

Debug pytest failures

@author: nbaya
"""

import os, subprocess, shutil
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D # only used to make legend
from matplotlib import collections  as mc
import matplotlib as mpl
import re # only used during plotting
from math import ceil, floor # only used for plotting

from COVID19.parameters import ParameterSet
from COVID19.model import VaccineSchedule

if os.path.isdir('/Users/nbaya'):
    OPENABM_DIR = '/Users/nbaya/gms/fraser_lab/OpenABM-Covid19'
    PLOTS_DIR = '/Users/nbaya/Downloads'
else:
    OPENABM_DIR = '/well/fraser/users/liu380/OpenABM-Covid19'
    PLOTS_DIR = '/well/fraser/users/liu380/OpenABM-Covid19-simulations/plots'

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

os.chdir(OPENABM_DIR)
import covid19
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
            n_total,
            end_time,
            mild_fraction_0_9,
            mild_fraction_10_19,
            mild_fraction_20_29,
            mild_fraction_30_39,
            mild_fraction_40_49,
            mild_fraction_50_59,
            mild_fraction_60_69,
            mild_fraction_70_79,
            mild_fraction_80,
            mild_infectious_factor  
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
        params.set_param( "end_time", end_time )
        params.set_param( "n_total", n_total )
        params.set_param( "mild_fraction_0_9", mild_fraction_0_9 )
        params.set_param( "mild_fraction_10_19", mild_fraction_10_19 )
        params.set_param( "mild_fraction_20_29", mild_fraction_20_29 )
        params.set_param( "mild_fraction_30_39", mild_fraction_30_39 )
        params.set_param( "mild_fraction_40_49", mild_fraction_40_49 )
        params.set_param( "mild_fraction_50_59", mild_fraction_50_59 )
        params.set_param( "mild_fraction_60_69", mild_fraction_60_69 )
        params.set_param( "mild_fraction_70_79", mild_fraction_70_79 )
        params.set_param( "mild_fraction_80", mild_fraction_80 )
        mild_infectious_factor_current = mild_infectious_factor[0]
        mean_total_infected_current = mean_total_infected(params, mild_infectious_factor_current)

        # calculate the total infections for the rest and compare with the current
        for idx in range(1, len(mild_infectious_factor)):
            mild_infectious_factor_new = mild_infectious_factor[idx]
            mean_total_infected_new = mean_total_infected(params, mild_infectious_factor_new)
            
            # check the total infections
            # if mild_infectious_factor_new > mild_infectious_factor_current:
            #     np.testing.assert_equal( mean_total_infected_new > mean_total_infected_current, True)
            # elif mild_infectious_factor_new < mild_infectious_factor_current:
            #     np.testing.assert_equal( mean_total_infected_new < mean_total_infected_current, True)
            # elif mild_infectious_factor_new == mild_infectious_factor_current:
            #     np.testing.assert_allclose( mean_total_infected_new, mean_total_infected_current, atol = 0.01)
            print(f'{mild_infectious_factor_current}\t\t{mild_infectious_factor_new}'+
                  f'\t\t{mean_total_infected_current}\t\t{mean_total_infected_new}')
            
            # refresh current values
            mild_infectious_factor_current = mild_infectious_factor_new
            mean_total_infected_current = mean_total_infected_new

def test_infectiousness_multiplier( self, test_params, sd_multipliers ):
    """
    Check that the total infected stays the same up to 0.5 SD.
    """    
    ordered_multipliers = sorted( sd_multipliers )
    transmissions = []
    total_infected_means = []
    for sd_multiplier in ordered_multipliers:
        total_infected = []
        for rng_seed in range(1,21):
            params = utils.get_params_swig()
            for param, value in test_params.items():
                params.set_param( param, value )  
            params.set_param( "sd_infectiousness_multiplier", sd_multiplier )
            params.set_param( "rng_seed", rng_seed )
            model  = utils.get_model_swig( params )
              
            for time in range( test_params[ "end_time" ] ):
                model.one_time_step()
              
            results = model.one_time_step_results()
            total_infected.append( results[ "total_infected" ] )
              
            del model
            del params

        total_infected_means.append(np.mean(total_infected))        
    
    # print('\t'.join(([str(total_infected_means[0])]*len(total_infected_means))))
    # print('\t'.join([str(x) for x in total_infected_means]))
    base_infected_mean = total_infected_means[0]
    np.testing.assert_allclose([base_infected_mean]*len(total_infected_means), total_infected_means, rtol=0.05)
        
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

def infect_with_strains(model, n_infections_per_strain, trans_mult, time_mult):
    if sum(n_infections_per_strain)==0: # if no infections, do nothing
        return
    n_strains = len(n_infections_per_strain)
    total_strain_infections = sum(n_infections_per_strain)
    
    model.write_individual_file()
    df_indiv = pd.read_csv( constant.TEST_INDIVIDUAL_FILE, comment="#", sep=",", skipinitialspace=True )
    idxs     = df_indiv[ df_indiv[ "current_status" ] == constant.EVENT_TYPES.SUSCEPTIBLE.value ]['ID'].to_numpy()
    n_susc   = len( idxs )
    inf_id = np.random.choice( n_susc, total_strain_infections, replace=False) # individuals to infect
    infect_params = [(strain_idx, trans_mult[strain_idx], time_mult[strain_idx]) 
                   for strain_idx in range(n_strains) 
                   for infection_idx in range(n_infections_per_strain[strain_idx])] # parameter tuples for each infection
    
    for ind_idx, (strain_idx, transmission_multiplier, time_multiplier) in enumerate(infect_params):
        model.seed_infect_by_idx( 
            ID = idxs[ inf_id[ ind_idx ] ], 
            strain_idx = strain_idx, 
            transmission_multiplier = transmission_multiplier,
            time_multiplier = time_multiplier
        )
        
def print_sim_id(test_params, mid_sim_infect_params = None, vaccine_params = None,
                 fname=False, **kwargs):
    test_params_str = ','.join([''.join([f[0] for f in k.split('_')])+':'+str(v) for k,v in test_params.items()])
    kwarg_params_str = ','.join([''.join([f[0] for f in k.split('_')])+':'+str(v) for k,v in kwargs.items()])
    strs_to_join = [test_params_str, kwarg_params_str]
    for other_params in [mid_sim_infect_params, vaccine_params]:
        if other_params is not None:
            other_params_str = ','.join([''.join([f[0] for f in k.split('_')])+':'+str(v) for k,v in other_params.items()])
            if not fname: 
                other_params_str = '\n'+other_params_str
            strs_to_join.append(other_params_str)            
    all_params_str = ','.join(strs_to_join)
    all_params_str = all_params_str.replace('range','')
    if fname:
        chars_to_remove = [':',',','\(','\)','\[','\]']
        all_params_str = re.sub('|'.join(chars_to_remove), "", all_params_str)
        all_params_str = all_params_str.replace(' ','-')
    else:
        all_params_str = all_params_str.replace(' ','')
    return all_params_str


def test_cross_immunity():
    
    def run_simulation( test_params, cross_immunity, trans_mult, time_mult,
                       n_seed_infections, n_strains, mutation_rate, transmission_multiplier_sigma, 
                       rng_seed_range, mid_sim_infect_params = None, vaccine_params = None):
        results_list = []
        infections_list = []
        strains_list = []
        distances_list = []
        print(cross_immunity, mutation_rate)
        for rng_seed in rng_seed_range:
            params = utils.get_params_swig()
            if test_params is not None:
                for param, value in test_params.items():
                    params.set_param( param, value )
            params.set_param( 'rng_seed', rng_seed )
            params.set_param( 'n_seed_infection', 0 ) # manually seed infections for each strain later
            model  = utils.get_model_swig( params )
            # model.set_cross_immunity_matrix(cross_immunity)
            # if rho is not None:
            #     covid19.set_up_cross_immunity_draws(model.c_model, rho)
            for strain_idx in range(0,n_strains): 
                try:
                    covid19.initialise_strain(
                        model.c_model, 
                        strain_idx, 
                        trans_mult[strain_idx],
                        time_mult[strain_idx]) # need to initialise all strains up front to be able to set recovery time using cross immunity
                except:
                    print('Failed to initialise strain{strain_idx}')
            for strain_idx1 in range(n_strains-1):
                for strain_idx2 in range(strain_idx1+1, n_strains):
                    distance = 1-cross_immunity[strain_idx1][strain_idx2]
                    covid19.set_antigen_phen_distance(
                        model.c_model, 
                        strain_idx1, 
                        strain_idx2, 
                        distance)
            covid19.set_model_param_mutation_rate(
                model.c_model, 
                mutation_rate
            )
            covid19.set_model_param_transmission_multiplier_sigma(
                model.c_model, 
                transmission_multiplier_sigma
            )
            np.random.seed(model.c_params.rng_seed)
            
            if vaccine_params is not None:
                vaccine_schedule = VaccineSchedule(**vaccine_params)
                model.vaccinate_schedule(vaccine_schedule)
            
            # try:
            
            infect_with_strains(model=model, 
                                n_infections_per_strain=n_seed_infections, 
                                trans_mult=trans_mult,
                                time_mult=time_mult)
            
            for time in range( model.c_params.end_time ):
                model.one_time_step()
                if mid_sim_infect_params is not None:
                    n_infections_per_strain = [n*(time>=start_time)*(time<=end_time) 
                                               for start_time, end_time, n in 
                                               zip(mid_sim_infect_params['infection_start_time'],
                                                   mid_sim_infect_params['infection_end_time'],
                                                   mid_sim_infect_params['n_infections_per_day'])]
                    
                    infect_with_strains(model=model, 
                                        n_infections_per_strain=n_infections_per_strain, 
                                        trans_mult=trans_mult,
                                        time_mult=time_mult)

            results_list.append(model.results)
            
            model.write_transmissions()
            df_trans = pd.read_csv( constant.TEST_TRANSMISSION_FILE, comment="#", sep=",", skipinitialspace=True )
            model.write_strains()
            df_strains = pd.read_csv( constant.TEST_STRAINS_FILE, comment="#", sep=",", skipinitialspace=True )
            model.write_antigen_phen_distances()
            df_dist = pd.read_csv( constant.TEST_ANTIGEN_PHEN_DISTANCE_FILE, comment="#", sep=",", skipinitialspace=True )
            n_strains_all = max(n_strains, df_strains.strain_idx.max()+1) # number of strains
            infected_per_strain_df = {f'strain{strain_idx}':[] for strain_idx in range(n_strains_all)}
            max_time = model.c_params.end_time #df_trans.time_infected.max()
            for time in range( max_time+1 ): # infection events can occur from day 0 to the last day
                # df_time = df_trans[(df_trans.time_infected<time) & (time<df_trans.time_susceptible)]
                df_time = df_trans[(df_trans.time_infected==time)]
                for strain_idx in range(n_strains_all):
                    infected_per_strain_df[f'strain{strain_idx}'].append((df_time.strain_idx==strain_idx).sum())
            infected_per_strain_df = pd.DataFrame(infected_per_strain_df)
            infected_per_strain_df['time'] = range( max_time+1 )
            
            infections_list.append(infected_per_strain_df)
            strains_list.append(df_strains)
            distances_list.append(df_dist)
            # except:
            #     print(f'failed rng_seed={rng_seed}, time={time}')
            del model, params
        
        return results_list, infections_list, strains_list, distances_list
    
    def run_vaccine_sim( test_params ):
        params = utils.get_params_swig()
        if test_params is not None:
            for param, value in test_params.items():
                params.set_param( param, value )
        model  = utils.get_model_swig( params )
        vaccine_params = dict(frac_50_59 = 0,
                              frac_60_69 = 0,
                              frac_70_79=1, 
                              frac_80 = 1,
                              efficacy = 0)
        
        for time in range( model.c_params.end_time ):
            model.one_time_step()
        for age_group in [k for k in vaccine_params if 'frac_' in k]:
            age_group = age_group.replace('frac_','')
            plt.plot(model.results.time, model.results[f'total_infected_{age_group}'], label=age_group)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('total_infected')
        del params, model
    
    def plot_scenarios(field, sims, sim_labels, test_params, 
                       mid_sim_infect_params, vaccine_params, agg=True, **kwargs ):
        plt.figure()
        for idx, (results_list, _, _, _) in enumerate(sims):
            if agg:
                results = pd.concat(results_list, axis=0)
                results_dict = {agg_fn:results.groupby(by='time').agg(agg_fn) for agg_fn in ['mean', 'sem']}
                x = results_dict['mean'].index
                mean = results_dict['mean'][field]
                sem = results_dict['sem'][field]
                plt.plot(x, mean, label=sim_labels[idx], c=COLORS[idx])
                plt.fill_between(x, y1=mean-1.96*sem, y2=mean+1.96*sem, color=COLORS[idx], alpha=0.1)
            else:
                plt.plot(results_list[0].time, pd.concat([results[field] for results in results_list],axis=1), 
                         c=COLORS[idx], lw=1, alpha=0.5)
        if agg:
            plt.legend()
        else:
            custom_lines = [Line2D([0], [0], color=COLORS[i], lw=2) for i in range(len(sims))]
            plt.legend(custom_lines, [sim_labels[i] for i in range(len(sims))])
        plt.xlabel('time')
        plt.ylabel(field)
        title_str = f'n_strains: {kwargs["n_strains"]}\n{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, **kwargs)}'
        plt.title(title_str, fontsize=10 if all([arg is None for arg in [mid_sim_infect_params, vaccine_params]]) else 5)
        fname = (f'cross_immunity_v4_{"-".join([sim_labels[i] for i in range(len(sims))])}.'+
                 field+f'.{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, fname=True, **kwargs)}{"" if agg else ".not_agg"}.png')
        plt.savefig(f'{PLOTS_DIR}/{fname}', dpi=300)
    
    def plot_infections_by_strain(infections_list, label, test_params, mid_sim_infect_params, 
                                  vaccine_params, agg=True, **kwargs):
        if agg:
            infections = pd.concat(infections_list, axis=0)
            infections_dict = {agg_fn:infections.groupby(by='time').agg(agg_fn) for agg_fn in ['mean', 'sem']}
        plt.figure()
        for strain_idx  in range(n_strains):
            if agg:
                x = infections_dict['mean'].index
                mean = infections_dict['mean'][f'strain{strain_idx}']
                sem = infections_dict['sem'][f'strain{strain_idx}']
                plt.plot(x, mean, label=f'strain{strain_idx}', c=COLORS[strain_idx])
                plt.fill_between(x, y1=mean-1.96*sem, y2=mean+1.96*sem, color=COLORS[strain_idx], alpha=0.1)
            else:
                 for rep_idx in range(len(infections_list)):#range(len(kwargs['rng_seed_range'])): # rep_idx is not the same as rng_seed
                     x = infections_list[rep_idx]['time']
                     y = infections_list[rep_idx][f'strain{strain_idx}']
                     plt.plot(x, y, c=COLORS[strain_idx], lw=1, alpha=0.5)
        if agg:
            plt.legend()
        else:
            custom_lines = [Line2D([0], [0], color=COLORS[i], lw=2) for i in range(n_strains)]
            plt.legend(custom_lines, [f'strain{i}' for i in range(n_strains)])
        plt.xlim([-1, max(50, test_params['end_time']-50)])
        plt.xlabel('time')
        plt.ylabel('infections caused by strain')
        title_str = f'{label}\n{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, **kwargs)}'
        plt.title(title_str, fontsize=10 if all([arg is None for arg in [mid_sim_infect_params, vaccine_params]]) else 5)
        fname = f'n_infected_per_strain_v4.{label}.{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, fname=True, **kwargs)}{"" if agg else ".not_agg"}.png'
        plt.savefig(f'{PLOTS_DIR}/{fname}', dpi=300)
        
    def stacked_bar_plot_infections(infections_list, label, test_params, mid_sim_infect_params, 
                                    vaccine_params, proportion=False, **kwargs):
        infections = pd.concat(infections_list, axis=0)
        mean_infections = infections.groupby(by='time').agg('mean') # take mean across replicates
        bottom = 0
        x = mean_infections.index
        if proportion:
            total_infections_per_day = mean_infections.sum(axis=1)
            mean_infections = mean_infections/total_infections_per_day[:,None]
            mean_infections = mean_infections.fillna(0)
        plt.figure()
        for strain_idx  in range(n_strains):
            y = mean_infections[f'strain{strain_idx}']
            plt.bar(x, y, bottom=bottom, label=f'strain{strain_idx}', color=COLORS[strain_idx], width=1)
            bottom += y
        plt.legend()
        plt.xlim([-1,test_params['end_time']-50])
        plt.xlabel('time')
        plt.ylabel(f'{"proportion of " if proportion else ""}infections caused by strains')
        title_str = f'{label}\n{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, **kwargs)}'
        plt.title(title_str, fontsize=10 if all([arg is None for arg in [mid_sim_infect_params, vaccine_params]]) else 5)
        fname = f'prop_infected_per_strain_v4.{label}.{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, fname=True, **kwargs)}{".proportion" if proportion else ""}.png'
        plt.savefig(f'{PLOTS_DIR}/{fname}', dpi=300)
        
    def bivariate_kde(field, sims, sim_labels, test_params, mid_sim_infect_params, 
                      vaccine_params, **kwargs ):
        nrows = ceil(len(sims)**(1/2))
        fig, axs = plt.subplots(nrows=nrows, ncols=nrows, sharey=True, sharex=True)
        
        cmaps = ['Blues', 'Oranges','Greens','Reds','Purples','Greys']
                
        for idx, (results_list, _, _, _) in enumerate(sims):
                ax = axs if len(sims)==1 else axs[floor(idx/nrows)][idx%nrows]
                sns.kdeplot(data=pd.concat([results['time'] for results in results_list],axis=0), 
                            data2=pd.concat([results[field] for results in results_list],axis=0),
                            levels=10,shade_lowest=False, ax=ax,
                            cmap=cmaps[idx], label=sim_labels[idx]
                            )
                ax.legend(loc='upper left')
                
        title_str = f'n_strains: {kwargs["n_strains"]}\n{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, **kwargs)}'
        plt.suptitle(title_str, fontsize=10 if all([arg is None for arg in [mid_sim_infect_params, vaccine_params]]) else 5)
        fname = (f'bivarkde_v4_{"-".join([sim_labels[i] for i in range(len(sims))])}.'+
                 field+f'.{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, fname=True, **kwargs)}.png')
        plt.savefig(f'{PLOTS_DIR}/{fname}', dpi=300)
        
    def strain_network(sims, sim_labels, test_params, mid_sim_infect_params, 
                       vaccine_params, rng_seed_range, **kwargs ):        
        if len(sims)>1:
            nrows = len(rng_seed_range)
            fig, axs = plt.subplots(nrows=nrows, ncols=len(sims), sharey=True, sharex=True,
                                    figsize = (3*len(sims), 3*nrows))
        else:
            nrows = ceil(len(rng_seed_range)**(1/2))
            fig, axs = plt.subplots(nrows=nrows, ncols=nrows, sharey=True, sharex=True,
                                    figsize = (3*nrows, 2*nrows))
        
        # cmaps = ['Blues', 'Oranges','Greens','Reds','Purples','Greys']
        get_size  = lambda n_infections: 10*np.log10(1+n_infections)
        use_color = False        
        for idx, (_, infections_list, strains_list, _) in enumerate(sims):
            for rep_idx in range(len(rng_seed_range)):
                if len(sims)>1:
                    ax = axs[rep_idx][idx]
                else:
                    ax = axs[floor(rep_idx/nrows)][rep_idx%nrows]
                x = strains_list[rep_idx]['antigen_phen_dim1']
                y = strains_list[rep_idx]['antigen_phen_dim2']
                lines = [[(x[parent_idx],y[parent_idx]), (x[idx], y[idx])] 
                          for idx, parent_idx in enumerate(strains_list[rep_idx]['parent_idx']) if parent_idx!=-1]
                
                lc = mc.LineCollection(lines, color='k', linewidths=1, alpha=0.1)
                # sns.kdeplot(data=x, data2=y, levels=10,shade_lowest=False, ax=ax,
                #             cmap=cmaps[idx], label=sim_labels[idx], alpha=0.5)
                ax.add_collection(lc)
                if all(strains_list[rep_idx].transmission_multiplier==1):
                    color = 'k'
                else:
                    color = strains_list[rep_idx].transmission_multiplier
                    use_color = True
                scatter = ax.scatter(x, y, s = get_size(infections_list[rep_idx].aggregate('sum')[:-1]), # get every row in agg table except last which contains "time"
                                     c=color, alpha=0.5, label=sim_labels[idx],
                                     cmap='RdYlGn_r', vmin=0, vmax=2)
                ax.plot(*[0,0], 'kx', ms=3)
                # ax.legend(loc='upper left')
        
        kw = dict(prop="sizes", num=[1,10,100,1000,10000], color='k', func=lambda s: 10**(s/10)-1)
        fig.legend(*scatter.legend_elements(**kw), fontsize=5, 
                    bbox_to_anchor=(0.95,0.5), loc = 'center right')
        plt.tight_layout(pad=0.5)
        if use_color:
            cbaxes = fig.add_axes([0.90, 0.1, 0.02, 0.15])
            norm = mpl.colors.Normalize(vmin=0, vmax=2)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='RdYlGn_r'),
                         cax=cbaxes)
            cbar = fig.colorbar(scatter, cax = cbaxes)
            cbar.ax.tick_params(labelsize=5)
        plt.subplots_adjust(bottom=0.1, top=0.92, left=0.1, right=0.88)
        fig.text(0.5, 0.04, 'antigen phen dim 1', ha='center')
        fig.text(0.04, 0.5, 'antigen phen dim 2', va='center', rotation='vertical')
        n_strains = kwargs["n_strains"] if len(sims)>1 else [sims[0][2][i].strain_idx.max() for i in range(len(rng_seed_range))]
        title_str = f'n_strains: {n_strains}\n{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, **kwargs)}'
        plt.suptitle(title_str, fontsize=10 if all([arg is None for arg in [mid_sim_infect_params, vaccine_params]]) else 5)
        fname = (f'strain_network{"-".join([sim_labels[i] for i in range(len(sims))])}.'+
                 f'{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, fname=True, **kwargs)}.png')
        plt.savefig(f'{PLOTS_DIR}/{fname}', dpi=300)
        
    def plot_per_rep(field, sims, sim_labels, test_params, mid_sim_infect_params, 
                       vaccine_params, rng_seed_range, **kwargs ):        
        if len(sims)>1:
            nrows = len(rng_seed_range)
            fig, axs = plt.subplots(nrows=nrows, ncols=len(sims), sharey=True, sharex=True,
                                    figsize = (3*len(sims), 3*nrows))
        else:
            nrows = ceil(len(rng_seed_range)**(1/2))
            fig, axs = plt.subplots(nrows=nrows, ncols=nrows, sharey=True, sharex=True,
                                    figsize = (3*nrows, 2*nrows))
           
        for idx, (results_list, _, _, _) in enumerate(sims):
            for rep_idx in range(len(rng_seed_range)):
                if len(sims)>1:
                    ax = axs[rep_idx][idx]
                else:
                    ax = axs[floor(rep_idx/nrows)][rep_idx%nrows]
                ax.plot(results_list[rep_idx].time, results_list[rep_idx][field], 
                         c=COLORS[idx], lw=1, alpha=0.5)
        # custom_lines = [Line2D([0], [0], color=COLORS[i], lw=2) for i in range(len(sims))]
        # plt.legend(custom_lines, [sim_labels[i] for i in range(len(sims))])
        plt.tight_layout(pad=0.5)
        plt.subplots_adjust(bottom=0.1, top=0.92, left=0.1, right=0.88)
        fig.text(0.5, 0.04, 'time', ha='center')
        fig.text(0.04, 0.5, field, va='center', rotation='vertical')
        n_strains = kwargs["n_strains"] if len(sims)>1 else [sims[0][2][i].strain_idx.max() for i in range(len(rng_seed_range))]
        title_str = f'n_strains: {n_strains}\n{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, **kwargs)}'
        plt.suptitle(title_str, fontsize=10 if all([arg is None for arg in [mid_sim_infect_params, vaccine_params]]) else 5)
        fname = (f'plot_per_rep.{field}.{"-".join([sim_labels[i] for i in range(len(sims))])}.'+
                 f'{print_sim_id(test_params, mid_sim_infect_params, vaccine_params, fname=True, **kwargs)}.png')
        plt.savefig(f'{PLOTS_DIR}/{fname}', dpi=300)
    
    # rho = 0
    # n_strains = 6
    mid_sim_infect_params = dict(infection_start_time  = [0,200], 
                                  infection_end_time   = [0,200], 
                                  n_infections_per_day = [0,10])
    # vaccine_params = dict(frac_50_59 = 1,
    #                       frac_60_69 = 1,
    #                       frac_70_79=1, 
    #                       frac_80 = 1,
    #                       efficacy = 1)

    # infections_list = sims[0][1][0]
    # strains_list = sims[0][2][0]
    # distances_list = sims[0][2][0]

    for n_strains in [2]:
        for trans_mult in [
                [1]*n_strains,
                [1]*(n_strains-1)+[1.5]
                # [1+0.5*strain_idx for strain_idx in range(n_strains)],
                # [1+1*strain_idx for strain_idx in range(n_strains)],
                ]:
            for time_mult in [
                    [1]*(n_strains-1)+[1.5],
                    [1]*n_strains,
                    ]:
                all_params = dict(test_params = dict(n_total = 10000,
                                                     infectious_rate = 2,
                                                     end_time = 450), # default infectious_rate: 5.18
                                  n_strains         = n_strains,
                                  n_seed_infections = [200,0],
                                  trans_mult        = trans_mult,
                                  time_mult         = None,
                                  mutation_rate     = 0,
                                  transmission_multiplier_sigma = 0,
                                  rng_seed_range    = range(1,11),
                                  mid_sim_infect_params = None,
                                  vaccine_params = None)
                full = np.ones(shape=(all_params['n_strains'],all_params['n_strains']))
                zero = np.identity(all_params['n_strains'])
                half = np.full(shape=(all_params['n_strains'],all_params['n_strains']), fill_value=0.5)
                const1 = np.full(shape=(all_params['n_strains'],all_params['n_strains']), fill_value=0.99)
                const2 = np.full(shape=(all_params['n_strains'],all_params['n_strains']), fill_value=0.9)
                np.fill_diagonal(half, 1)
                np.fill_diagonal(const1, 1)
                np.fill_diagonal(const2, 1)
                cross_immunity_list = [full,
                                       const1,
                                        # half,
                                       const2,
                                       zero]
                # cross_immunity_list = [full,
                #                         np.triu(full, 0),# half_prob,
                #                         np.tril(full, 0),
                #                         zero]
                # cross_immunity_list = [full,
                #                         half,
                #                         zero,
                #                         [[1, 0.5, 0.4],
                #                          [0.5, 1, 0.4],
                #                          [0.5, 0.5, 1]],
                #                         [[1, 0.5, 0.75],
                #                          [0.5, 1, 0.75],
                #                          [0.5, 0.5, 1]]]
                # cross_immunity_list = [full,
                #                         half, 
                #                         zero, 
                #                         [[round(1-0.1*abs(i-j),1) for j in range(n_strains)] 
                #                         for i in range(n_strains)],
                #                         [[round(1-0.2*abs(i-j),1) for j in range(n_strains)] 
                #                         for i in range(n_strains)]]
                # cross_immunity_list = [[[round(1-0.025*abs(i-j),1) for j in range(n_strains)] 
                #                         for i in range(n_strains)],
                #                         [[round(1-0.05*abs(i-j),1) for j in range(n_strains)] 
                #                         for i in range(n_strains)],
                #                         [[round(1-0.1*abs(i-j),1) for j in range(n_strains)] 
                #                         for i in range(n_strains)],
                #                         ]
                sim_labels = ['full',
                              '0.99',
                              # 'half', 
                              '0.90',
                              'zero']
                # sim_labels = ['full', 'tri upper', 'tri lower', 'zero']
                # sim_labels = ['full','half', 'zero', 'diverge0.1', 'diverge0.2']
                # sim_labels = ['diverge0.025', 'diverge0.05', 'diverge0.1']
                # rho_list = [0,0.1,0.5,1]
                # sim_labels = [f'half-rho{rho}' for rho in rho_list]
                # cross_immunity_list = [half for _ in rho_list]
            # for rho in rho_list:
                
                if n_strains==1:
                    cross_immunity_list = cross_immunity_list[:1]
                    sim_labels=sim_labels[:1]
                
                sims = []
                for idx, cross_immunity in enumerate(cross_immunity_list):
                    results_list, infections_list, strains_list, distances_list = run_simulation(cross_immunity=cross_immunity, 
                                                                                 **all_params)
                    sims.append((results_list, infections_list, strains_list, distances_list))
                    plot_infections_by_strain(infections_list=infections_list,
                                              label=sim_labels[idx],
                                              **all_params)
                    plot_infections_by_strain(infections_list=infections_list,
                                              label=sim_labels[idx],
                                              agg=False,
                                              **all_params)
                    stacked_bar_plot_infections(infections_list=infections_list,
                                                label=sim_labels[idx],
                                                **all_params)
                    stacked_bar_plot_infections(infections_list=infections_list,
                                                label=sim_labels[idx],
                                                proportion=True,
                                                **all_params)
                    plt.close('all')
                
                # strain_network(sims=sims, 
                #                sim_labels=sim_labels, 
                #                **all_params)
                
                for field in ['total_infected']:
                    plot_scenarios(field=field, 
                                   sims=sims, 
                                   sim_labels=sim_labels,
                                   **all_params)
                    plot_scenarios(field=field, 
                                   sims=sims, 
                                   sim_labels=sim_labels,
                                   agg=False,
                                   **all_params)
                    bivariate_kde(field=field, 
                                  sims=sims, 
                                  sim_labels=sim_labels, 
                                  **all_params)
                    # plot_per_rep(field=field, 
                    #              sims=sims, 
                    #              sim_labels=sim_labels, 
                    #              **all_params)
                    plt.close('all')

def test_corr_cross_immunity():
    def corr_cross_immunity_sim( rho, cross_immunity_p, n_strains, 
                                test_params = dict(rng_seed = 1, n_total = 10000, end_time = 50)):
        params = utils.get_params_swig()
        if test_params is not None:
            for param, value in test_params.items():
                params.set_param( param, value )  
        model  = utils.get_model_swig( params )
        # covid19.set_up_cross_immunity_draws(model.c_model, rho)
        cross_immunity = np.full(shape=(n_strains,n_strains), fill_value=cross_immunity_p)
        np.fill_diagonal(cross_immunity, 1)
        model.set_cross_immunity_matrix(cross_immunity=cross_immunity)
        infect_with_strains(model=model, 
                            n_infections_per_strain=[round(120/n_strains)]*n_strains, 
                            trans_mult=[1]*n_strains)
        for _ in range( model.c_params.end_time ):
            model.one_time_step()
        # print(f'rho: {rho}, cross_imm p: {cross_immunity_p}, total_infected: {model.results.total_infected.values[-1]}\n')
        return model.results.total_infected.values[-1]
        del model, params
    
    def plot_heatmap( data, title, xticklabels, yticklabels, xlabel, ylabel, cbar_label, test_params, **kwargs):
        plt.figure()
        sns.heatmap(data, yticklabels = yticklabels, xticklabels = xticklabels,
                    cbar_kws = {'label': cbar_label})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(print_sim_id(test_params, **kwargs))
        fname = f'heatmap.{cbar_label.replace(" ","_")}.{print_sim_id(test_params, fname=True, **kwargs)}.png'
        plt.savefig(f'{PLOTS_DIR}/{fname}',dpi=300)
        
    
    rng_seed_range = range(1,51)
    n_strains_list = [60,40,30,20,15,10,5,2]
    cross_immunity_p_list = [0.95] #np.round(np.linspace(0,1,3),2)
    rho_list = np.round(np.linspace(0,1,11),2)
    test_params = dict(n_total = 10000, 
                       end_time = 50,
                       infectious_rate = 5)
    for cross_immunity_p in cross_immunity_p_list:
        print(f'cross_immunity: {cross_immunity_p}')
        total_infected_mean = np.zeros(shape=(len(n_strains_list), len(rho_list)))
        total_infected_std = np.zeros(shape=(len(n_strains_list), len(rho_list)))
        for i, n_strains in enumerate(n_strains_list):
            print(f'n_strains: {n_strains}')
            for j, rho in enumerate(rho_list):
                print(f'rho: {rho}')
                total_infected = []
                for rng_seed in rng_seed_range:
                    test_params_copy = test_params.copy()
                    test_params_copy['rng_seed'] = rng_seed
                    try:
                        total_infected.append(corr_cross_immunity_sim(rho=rho, 
                                                                      cross_immunity_p=cross_immunity_p,
                                                                      n_strains = n_strains,
                                                                      test_params=test_params,
                                                                      ))
                    except:
                        print(f'Failed rng_seed: {rng_seed}')
                total_infected_mean[i][j] = np.mean(total_infected)
                total_infected_std[i][j] = np.std(total_infected)
        
        all_params = dict(test_params       = test_params,
                          rng_seed_range    = range(1,51),
                          cross_immunity_p  = cross_immunity_p)
        
        plot_heatmap(data=total_infected_mean, title=f'cross-immunity prob. = {cross_immunity_p}', 
                     xticklabels=rho_list, yticklabels=n_strains_list, xlabel='rho', 
                     ylabel='n_strains', cbar_label='mean final total_infected', 
                     **all_params)
        plot_heatmap(data=total_infected_std, title=f'cross-immunity prob. = {cross_immunity_p}', 
                     xticklabels=rho_list, yticklabels=n_strains_list, xlabel='rho', 
                     ylabel='n_strains', cbar_label='stdev final total_infected', 
                     **all_params)

def main():
    destroy()
    set_up()
    
    test_cross_immunity()
    # kwargs = test_infection_dynamics.TestClass.params['test_monoton_mild_infectious_factor'][0].copy()
    # test_monoton_mild_infectious_factor(self=test_infection_dynamics.TestClass, **kwargs)
    
    # kwargs = test_infection_dynamics.TestClass.params['test_infectiousness_multiplier'][0].copy()
    # test_infectiousness_multiplier(self=test_infection_dynamics.TestClass, **kwargs)
    
    # kwargs = test_network.TestClass.params['test_custom_occupation_network'].copy()
    # test_params_list = [x['test_params'] for x in kwargs]
    # for rng_seed in range(1,21):
    #     print(f'rng_seed: {rng_seed}')
    #     # for idx, test_params in enumerate(test_params_list, 1):
    #     test_params = test_params_list[-1]
    #     test_params['rng_seed'] = rng_seed
    #     test_custom_occupation_network(self=test_network.TestClass, test_params=test_params)        
    
        
    destroy()

if __name__=='__main__':
    main()
