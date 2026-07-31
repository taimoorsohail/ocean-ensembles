#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=1
#SBATCH --threads-per-core=1
#SBATCH --exclusive
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

export JULIA_CUDA_MEMORY_POOL=none
export LD_LIBRARY_PATH=/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/OpenMPI/4.1.4/lib/:$LD_LIBRARY_PATH
export JULIA_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd ~/uom/ocean-ensembles/experiments/submission_scripts/

# julia -e 'using Pkg; Pkg.add("CUDA"); using CUDA; CUDA.set_runtime_version!(local_toolkit=true)'
# julia -e 'using Pkg; Pkg.add("MPIPreferences"); using MPIPreferences; use_system_binary()'

# julia --project -e 'using CUDA; CUDA.precompile_runtime()'
# julia --project -e 'using Pkg; Pkg.status()'

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