#!/bin/bash
#SBATCH -p gpu-h100
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --job-name=RYF_twelfth_test
#SBATCH --output=/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/outputs/RYF_twelfth/slurm-%j.out
#
# First 1/12° RYF test on one node of 4 H100s, Partition(2, 2): 30 days from ECCO at Δt = 10 min.
# Checks stability, water conservation and the combined Zarr output.
set -u
# Same MPI setup as interactive_sxtdeg.sh and tmpi: Julia's own OpenMPI_jll, started through the
# ./mpirun wrapper, which sources jll_runtime_env.sh to clear Spartan's system MPI settings.
# The first test (job 35606872) skipped the wrapper and crashed in the first halo exchange.
# squeue is not on the reduced PATH below, so read the job end time first
export JOB_END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)
module load Julia/1.12.6
julia_bin=$(command -v julia)
module purge
export PATH="$(dirname "$julia_bin"):/usr/local/bin:/usr/bin:/bin"
cd /data/gpfs/projects/punim2499/taimoor/ocean-ensembles/experiments/submission_scripts
export CELLS_PER_DEGREE=12 RUN_NAME=RYF_twelfth OUTPUT_ROOT=/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/outputs
export CHILD_ARCH=GPU DT_MINUTES=10 STOP_DAYS=30
# Quick check that GPU arrays can be passed between ranks before building the model
./mpirun --oversubscribe --bind-to none -n 4 "$julia_bin" --project gpu_mpi_test.jl || exit 1
./mpirun --oversubscribe --bind-to none -n 4 "$julia_bin" --project run_distributed_segment.jl
