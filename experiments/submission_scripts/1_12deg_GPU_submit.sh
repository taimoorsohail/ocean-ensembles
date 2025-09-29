#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=150G
#SBATCH --job-name=GPU_RYF12dg
#SBATCH --output=../run_logs/GPU_RYF1_12dg_%j.o
#SBATCH --error=../run_logs/GPU_RYF1_12dg_%j.e
#SBATCH --export=ALL

# === Setup resubmission ===
script_name="1_12deg_GPU_submit.sh"

# Default count/max
count=${count:-1}
max=${max:-$count}

echo "Run $count of $max"

target=$((count * 1))

# Run Julia
mpirun -n 4 julia --project ../RYF_twfdeg.jl --arch GPU --stop_time $target \
    > ../run_logs/GPU_RYF1_12dg_${count}.stdout \
    2> ../run_logs/GPU_RYF1_12dg_${count}.stderr

((count++))

# Resubmit if needed
if [ $count -le $max ]; then
    echo "Resubmitting model"
    sbatch --export=count=$count,max=$max $script_name
else
    echo "Last submission; $count of $max"
fi
