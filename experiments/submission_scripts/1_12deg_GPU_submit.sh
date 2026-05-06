#!/bin/bash
#PBS -P v46
#PBS -q gpuvolta
#PBS -l walltime=12:00:00
#PBS -l mem=150GB
#PBS -l storage=gdata/v46+gdata/hh5+gdata/e14+scratch/v46+scratch/v45+scratch/e14
#PBS -l wd
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l jobfs=10GB
#PBS -W umask=027
#PBS -j n 
#PBS -N GPU_RYF1_12dg

# Output logs
#PBS -o /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_12dg.o
#PBS -e /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_12dg.e

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

mpirun -n 8 julia --project \
  ../RYF_twfdeg.jl --arch GPU --stop_time $target\
  > /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_12dg_$count.stdout \
  2> /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_12dg_$count.stderr


((count++))

if [ $count -le $max ]; then
    echo "Resubmitting model"
    cd $PBS_O_WORKDIR
    qsub -v count=$count,max=$max $script_name
else
    echo "Last submission; $count of $max"
fi
