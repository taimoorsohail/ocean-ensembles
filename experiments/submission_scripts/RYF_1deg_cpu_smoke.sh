#!/bin/bash
#SBATCH -p sapphire
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --job-name=RYF_1deg_smoke
#SBATCH --output=/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/outputs/RYF_1deg_smoke/slurm-%j.out
#
# Smoke test of the distributed setup on 4 CPU ranks at 1°: 6 days at Δt = 30 min, so one
# 5-day average lands in every Zarr store. Delete outputs/RYF_1deg_smoke/ afterwards.
set -u
module load Julia/1.12.6
cd /data/gpfs/projects/punim2499/taimoor/ocean-ensembles/experiments/submission_scripts
export CELLS_PER_DEGREE=1 RUN_NAME=RYF_1deg_smoke OUTPUT_ROOT=/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/outputs
export CHILD_ARCH=CPU DT_MINUTES=30 STOP_DAYS=6 JULIA_NUM_THREADS=4 SEA_ICE_SUBSTEPS=40
export JOB_END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)
julia --project -e 'using MPI; run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project run_distributed_segment.jl`)'
