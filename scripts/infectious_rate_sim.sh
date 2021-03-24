#!/usr/bin/env bash
#
# Run OpenABM-Covid19 using the default parameters and adjusted `infectious_rate`
#
# Seed is 1 by default. If SGE_TASK_ID != "undefined", seed=SGE_TASK_ID
#
# Author: Nik Baya (2021-03-22)
#
#$ -N infect_rate_sim
#$ -wd /users/gms/whv244/fraser_lab/OpenABM-Covid19-simulations
#$ -o logs/infect_rate_sim.log
#$ -e logs/infect_rate_sim.errors.log
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
readonly infectious_rate=5.18 # baseline: 5.18
readonly params="1seed_p0.01_N1_0.1"
readonly out_prefix="infectrate${infectious_rate}_${params}" # 1 seed strain, mutation prob=0.01, transmission multiplier drawn from normal with mean=1, stdev=0.1
readonly out_dir="data/${out_prefix}.seed_${seed}"
readonly out="${out_dir}/${out_prefix}.seed_${seed}.csv" #stdout from simulation
readonly out_strain_info="${out_dir}/${out_prefix}.strain_info.seed_${seed}.tsv"

# executable path
readonly openabm_exe="${openabm_dir}/src/covid19ibm.default_${params}.exe"

get_strain_info() {
}


# run simulation
if [ ! -f ${out} ]; then
  mkdir -p ${out_dir}
  ${openabm_exe} \
    <( sed "s/^1/${seed}/g" "${openabm_dir}/tests/data/baseline_parameters.csv" | sed "s/5.18/${infectious_rate}/g" ) 1 \
    ${out_dir} \
    "${openabm_dir}/tests/data/baseline_household_demographics.csv" > ${out}
fi

if [ -f ${out} ] && [ ! -f ${out_strain_info} ]; then
  echo -e "total_strains\tmean_transmission_multiplier" > ${out_strain_info}
  paste <( grep "^# Total strains" ${out} | cut -f2 ) \
      <( grep "^# Mean t_mul" ${out} | cut -f3 )  >> ${out_strain_info}
fi
