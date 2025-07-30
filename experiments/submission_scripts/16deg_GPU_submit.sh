#!/bin/bash
#PBS -P v46
#PBS -q gpuvolta
#PBS -l walltime=6:00:00
#PBS -l mem=150GB
#PBS -l storage=gdata/v46+gdata/hh5+gdata/e14+scratch/v46+scratch/v45+scratch/e14
#PBS -l wd
#PBS -l ncpus=36 
#PBS -l ngpus=3 
#PBS -l jobfs=10GB
#PBS -W umask=027
#PBS -j n 
#PBS -N GPU_RYFsxthdg

# Output logs
#PBS -o /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYFsxthdg.o
#PBS -e /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYFsxthdg.e

# === Setup resubmission ===
script_name='16deg_GPU_submit.sh'

# Set default values of count and max
if [ -z $count ]; then
    count=1
fi

if [ -z $max ]; then
    max=$count
fi

# Log submission counters
echo "Run $count of $max"

mpirun -n 3 julia --project \
  ../RYF_sxthdeg.jl --arch GPU \
  > /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYFsxthdg_$count.stdout \
  2> /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYFsxthdg_$count.stderr


((count++))

if [ $count -le $max ]; then
    echo "Resubmitting model"
    cd $PBS_O_WORKDIR
    qsub -v count=$count,max=$max $script_name
else
    echo "Last submission; $count of $max"
fi
