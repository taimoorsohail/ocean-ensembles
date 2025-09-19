#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --job-name=GPU_RYF1_4dg
#SBATCH --output=/g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_4dg_%j.o
#SBATCH --error=/g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_4dg_%j.e
#SBATCH --export=ALL

# === Setup resubmission ===
script_name="1_4deg_GPU_submit.sh"

# Default count/max
count=${count:-1}
max=${max:-$count}

echo "Run $count of $max"

target=$((count * 4))

# Run Julia
julia --project ../RYF_qtrdeg.jl --arch GPU --stop_time $target \
    > /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_4dg_${count}.stdout \
    2> /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1_4dg_${count}.stderr

((count++))

# Resubmit if needed
if [ $count -le $max ]; then
    echo "Resubmitting model"
    sbatch --export=count=$count,max=$max $script_name
else
    echo "Last submission; $count of $max"
fi


