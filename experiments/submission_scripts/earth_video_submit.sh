#!/bin/bash
#PBS -P v46
#PBS -q normalbw
#PBS -l walltime=11:00:00
#PBS -l mem=150GB
#PBS -l storage=gdata/v46+gdata/e14+scratch/v46+scratch/v45+scratch/e14
#PBS -l wd
#PBS -l ncpus=1
#PBS -l jobfs=10GB
#PBS -W umask=027
#PBS -j n 
#PBS -N earth_video

# Output logs
#PBS -o /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/earth_video.o
#PBS -e /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/earth_video.e

julia --project \
  ../../analysis/earth_video.jl > /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/earth_video.stdout \
  2> /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/earth_video.stderr


