#!/bin/bash
#PBS -P v46
#PBS -q gpuvolta
#PBS -l walltime=01:00:00
#PBS -l mem=150GB
#PBS -l storage=gdata/v46+gdata/hh5+gdata/e14+scratch/v46+scratch/v45+scratch/e14
#PBS -l wd
#PBS -l ncpus=36 
#PBS -l ngpus=3 
#PBS -l jobfs=10GB
#PBS -W umask=027
#PBS -j n 
#PBS -N GPU_testoutputs

# Output logs
#PBS -o /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_test.o
#PBS -e /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_test.e

mpirun -n 3 julia --project \
  ../../test/test_Clmocn_outputs.jl \
  > /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_test_outputs.stdout \
  2> /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_test_outputs.stderr
