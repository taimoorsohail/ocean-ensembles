#!/bin/bash

#PBS -P v46 
#PBS -N testing_MPI
#PBS -q gpuvolta
#PBS -l walltime=1:00:00
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=384GB
#PBS -l jobfs=10GB
#PBS -W umask=027
#PBS -j n 
#PBS -l storage=gdata/v46+gdata/hh5+gdata/e14+scratch/v46+scratch/v45+scratch/e14
#PBS -l wd


# Use moar processes for precompilation to speed things up
export JULIA_NUM_PRECOMPILE_TASKS=48
export JULIA_NUM_THREADS=48

# Load critical modules
module --force purge
module load cuda openmpi #cray-mpich ncarenv nvhpc

export LD_LIBRARY_PATH=$(dirname $(which mpirun))/../lib:$LD_LIBRARY_PATH
# Utter mystical incantations to perform various miracles
export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_MPI_HAS_CUDA=true
export PALS_TRANSFER=false
export JULIA_CUDA_MEMORY_POOL=none

# Write down a script that binds MPI processes to GPUs (taken from Derecho documentation)
cat > launch.sh << EoF_s
#! /bin/bash

export MPICH_GPU_SUPPORT_ENABLED=1
export LOCAL_RANK=\${PMI_LOCAL_RANK}
export GLOBAL_RANK=\${PMI_RANK}
export CUDA_VISIBLE_DEVICES=\$(expr \${LOCAL_RANK} % 4)

echo "Global Rank \${GLOBAL_RANK} / Local Rank \${LOCAL_RANK} / CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES} / \$(hostname)"

exec \$*
EoF_s

chmod +x launch.sh

# Now to make our julia environment work:
# 1. Instantiate (we only need to do this once, but this also may be the first time you are running this code)
julia --project -e 'using Pkg; Pkg.instantiate()'
# 2. Add some packages to the environment that we need to use
# julia --project -e 'using Pkg; Pkg.add("MPI"); Pkg.add("MPIPreferences"); Pkg.add("CUDA"); Pkg.add("Oceananigans")'
# 3. Tell MPI that we would like to use the system binary we loaded with module load cray-mpich
julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
# 4. Build MPI and CUDA in advance for yucks
julia --project -e 'using MPI; using CUDA; CUDA.precompile_runtime()'

# Finally, let's run this thing
mpiexec -n 4 --map-by ppr:4:node ./launch.sh julia --project test_interpolate.jl
