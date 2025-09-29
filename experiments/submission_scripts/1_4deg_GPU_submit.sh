#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
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

<<<<<<< HEAD
target=$((count * 1))

# Run Julia
mpirun -np 4 julia --project ../RYF_qtrdeg.jl --arch GPU --stop_time $target \
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
