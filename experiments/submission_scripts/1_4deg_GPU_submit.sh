#!/bin/bash
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

cd ~/uom/ocean-ensembles/experiments/submission_scripts/

# Run Julia
mpirun -np 2 julia --project ../RYF_qtrdeg.jl --arch GPU --stop_time $target \
> ../run_logs/GPU_RYF1_4dg_${count}.stdout \
2> ../run_logs/GPU_RYF1_4dg_${count}.stderr

((count++))

# Resubmit if needed
if [ $count -le $max ]; then
echo "Resubmitting model"
sbatch --export=count=$count,max=$max $script_name
else
echo "Last submission; $count of $max"
fi
