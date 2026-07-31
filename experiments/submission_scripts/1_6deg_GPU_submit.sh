#!/bin/bash
<<<<<<<< HEAD:experiments/submission_scripts/1_4deg_GPU_submit.sh
#SBATCH --partition=gpu-a100
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --mem=150G
#SBATCH --job-name=GPU_RYF25dg
#SBATCH --output=../run_logs/GPU_RYF1_4dg_%j.o
#SBATCH --error=../run_logs/GPU_RYF1_4dg_%j.e
#SBATCH --export=ALL

# === Setup resubmission ===
script_name="1_4deg_GPU_submit.sh"
========
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
#PBS -N GPU_RYF1_6dg

# Output logs
#PBS -o /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_6dg.o
#PBS -e /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_6dg.e

# === Setup resubmission ===
script_name='1_6deg_GPU_submit.sh'
>>>>>>>> ts/parallel-run:experiments/submission_scripts/1_6deg_GPU_submit.sh

# Default count/max
count=${count:-1}
max=${max:-$count}

echo "Run $count of $max"
target=$((count * 1))

# === Load modules ===
module load foss/2022a UCX-CUDA/1.16.0-CUDA-12.4.1 Julia/1.10.8
module unload CUDA
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_NUM_THREADS=1

<<<<<<<< HEAD:experiments/submission_scripts/1_4deg_GPU_submit.sh
cd ~/uom/ocean-ensembles/experiments/submission_scripts/

# Run Julia
mpirun -np 2 julia --project ../RYF_qtrdeg.jl --arch GPU --stop_time $target \
> ../run_logs/GPU_RYF1_4dg_${count}.stdout \
2> ../run_logs/GPU_RYF1_4dg_${count}.stderr
========
mpi_args=""

mpiexec --report-bindings --bind-to socket --map-by socket -n 4 julia --project \
  ../RYF_sxtdeg.jl --arch GPU --stop_time $target\
  > /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_6dg_$count.stdout \
  2> /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_6dg_$count.stderr
>>>>>>>> ts/parallel-run:experiments/submission_scripts/1_6deg_GPU_submit.sh

((count++))

# Resubmit if needed
if [ $count -le $max ]; then
<<<<<<<< HEAD:experiments/submission_scripts/1_4deg_GPU_submit.sh
echo "Resubmitting model"
sbatch --export=count=$count,max=$max $script_name
========
    echo "Resubmitting model"
    cd $PBS_O_WORKDIR
    qsub -v count=$count, max=$max $script_name
>>>>>>>> ts/parallel-run:experiments/submission_scripts/1_6deg_GPU_submit.sh
else
echo "Last submission; $count of $max"
fi
