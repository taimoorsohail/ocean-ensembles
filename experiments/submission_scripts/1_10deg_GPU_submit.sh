#!/bin/bash
#PBS -P v46
#PBS -q dgxa100
#PBS -l walltime=6:00:00
#PBS -l mem=150GB
#PBS -l storage=gdata/v46+gdata/hh5+gdata/e14+scratch/v46+scratch/v45+scratch/e14
#PBS -l wd
#PBS -l ncpus=16 
#PBS -l ngpus=1
#PBS -l jobfs=10GB
#PBS -W umask=027
#PBS -j n 
#PBS -N GPU_RYF1_10dg

# Output logs
#PBS -o /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_10dg.o
#PBS -e /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_10dg.e

# === Setup resubmission ===
script_name='1_10deg_GPU_submit.sh'

# Set default values of count and max
if [ -z $count ]; then
    count=1
fi

if [ -z $max ]; then
    max=$count
fi

# Log submission counters
echo "Run $count of $max"

target=$((count * 4))  

julia --project \
  ../RYF_tntdeg.jl --arch GPU --stop_time $target\
  > /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_10dg_$count.stdout \
  2> /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_10dg_$count.stderr


((count++))

if [ $count -le $max ]; then
    echo "Resubmitting model"
    cd $PBS_O_WORKDIR
    qsub -v count=$count,max=$max $script_name
else
    echo "Last submission; $count of $max"
fi
