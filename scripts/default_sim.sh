#!/usr/bin/env bash
#
# Run OpenABM-Covid19 using the default parameters
#
# Seed is 1 by default. If SGE_TASK_ID != "undefined", seed=SGE_TASK_ID
#
# Author: Nik Baya (2021-03-18)
#
#$ -N default_sim
#$ -wd /users/gms/whv244/fraser_lab/OpenABM-Covid19-simulations
#$ -o logs/default_sim.log
#$ -e logs/default_sim.errors.log
#$ -q short.qe
#$ -V

set -o errexit
set -o nounset

if [ "${SGE_TASK_ID}" = "undefined" ]; then
    readonly seed=1
else
    readonly seed=${SGE_TASK_ID}
fi

# directories
readonly openabm_dir="/users/gms/whv244/fraser_lab/OpenABM-Covid19"

# output path
#readonly out_prefix="default_sim" # original simulation, single strain, no mutations
#readonly out_prefix="default_sim_transmul1.1" # original simulation, single strain, transmission multiplier = 1.1, no mutations
#readonly out_prefix="default_sim_transmul1.5" # original simulation, single strain, transmission multiplier = 1.5, no mutations
#readonly out_prefix="default_5seedstrains_transmul1" # every seed infection has a separate strain, but all strains have transmission multiplier = 1, no mutations
#readonly out_prefix="default_1seed_p0.0001_N1_0.001" # 1 seed strain, mutation prob=0.0001, transmission multiplier drawn from normal with mean=1, stdev=0.001
#readonly out_prefix="default_1seed_p0.0001_N1_0.01" # 1 seed strain, mutation prob=0.0001, transmission multiplier drawn from normal with mean=1, stdev=0.01
#readonly out_prefix="default_1seed_p0.0001_N1_0.1" # 1 seed strain, mutation prob=0.0001, transmission multiplier drawn from normal with mean=1, stdev=0.1
#readonly out_prefix="default_1seed_p0.0001_Nparent_0.1" # 1 seed strain, mutation prob=0.0001, transmission multiplier drawn from normal with mean=parent transmission multiplier, stdev=0.1 and cannot be negative
#readonly out_prefix="default_1seed_p0.0001_N1.1_0.1" # 1 seed strain, mutation prob=0.0001, transmission multiplier drawn from normal with mean=1.1, stdev=0.1
#readonly out_prefix="default_1seed_p0.001_N1_0.1" # 1 seed strain, mutation prob=0.001, transmission multiplier drawn from normal with mean=1, stdev=0.1
readonly out_prefix="default_1seed_p0.01_N1_0.1" # 1 seed strain, mutation prob=0.01, transmission multiplier drawn from normal with mean=1, stdev=0.1
#readonly out_prefix="default_1seed_p0.001_Nparent_0.1" # 1 seed strain, mutation prob=0.001, transmission multiplier drawn from normal with mean=parent transmission multiplier, stdev=0.1 and cannot be negative

readonly out="data/${out_prefix}.seed_${seed}.csv"
readonly out_strain_info="data/${out_prefix}.strain_info.seed_${seed}.tsv"

# executable path
readonly openabm_exe="${openabm_dir}/src/covid19ibm.${out_prefix}.exe"

# run simulation
if [ ! -f ${out} ]; then
  ${openabm_exe} \
          <(  sed "s/^1/${seed}/g" "${openabm_dir}/tests/data/baseline_parameters.csv" ) 1 \
          "dummy_dir" \
          "${openabm_dir}/tests/data/baseline_household_demographics.csv" > ${out}
fi

if [ -f ${out} ] && [ ! -f ${out_strain_info} ]; then
    echo -e "total_strains\tmean_transmission_multiplier" > ${out_strain_info}
    paste <( grep "^# Total strains" ${out} | cut -f2 ) \
        <( grep "^# Mean t_mul" ${out} | cut -f3 )  >> ${out_strain_info}
fi
