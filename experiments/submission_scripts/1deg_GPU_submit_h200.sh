#!/bin/bash
#PBS -P v46
#PBS -q gpuhopper
#PBS -l walltime=5:00:00
#PBS -l mem=150GB
#PBS -l storage=gdata/v46+gdata/hh5+gdata/e14+scratch/v46+scratch/v45+scratch/e14
#PBS -l wd
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l jobfs=10GB
#PBS -W umask=027
#PBS -j n 
#PBS -N GPU_RYF1dg

# Output logs
#PBS -o /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1dgh200.o
#PBS -e /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1dgh200.e

# === Setup resubmission ===
script_name='1deg_GPU_submit_h200.sh'

# Set default values of count and max
if [ -z $count ]; then
    count=1
fi

if [ -z $max ]; then
    max=$count
fi

# Log submission countersq
echo "Run $count of $max"

mpi_args=""

julia --project ../RYF_onedeg.jl --arch GPU --time $count\
  > /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF_1dg_h200_$count.stdout \
  2> /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF_1dg_h200_$count.stderr

((count++))

if [ $count -le $max ]; then
    echo "Resubmitting model"
    cd $PBS_O_WORKDIR
    qsub -v count=$count,max=$max $script_name
else
    echo "Last submission; $count of $max"
fi
