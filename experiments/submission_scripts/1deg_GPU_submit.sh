#!/bin/bash
#SBATCH --partition=gpu-a100       # or gpu-a100 if you specifically want A100 GPUs
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --job-name=GPU_RYF1dg
#SBATCH --output=/g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1dg_%j.o
#SBATCH --error=/g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1dg_%j.e
#SBATCH --chdir=/g/data/v46/txs156/ocean-ensembles/experiments/run_logs
#SBATCH --export=ALL

# === Setup resubmission ===
script_name="1deg_GPU_submit.sh"

# Set default values of count and max
count=${count:-1}
max=${max:-$count}

# Log submission counters
echo "Run $count of $max"

target=$((count * 7))

# Run Julia
julia --project ../RYF_onedeg.jl --arch GPU --stop_time $target \
    > /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1dg_${count}.stdout \
    2> /g/data/v46/txs156/ocean-ensembles/experiments/run_logs/GPU_RYF1dg_${count}.stderr

# Increment count
((count++))

# Resubmit if needed
if [ $count -le $max ]; then
    echo "Resubmitting model"
    sbatch --export=count=$count,max=$max $script_name
else
    echo "Last submission; $count of $max"
fi
