#!/usr/bin/env bash
#PBS -q expressbw
#PBS -P fy29
#PBS -l walltime=00:15:00
#PBS -l ncpus=1
#PBS -l mem=10GB
#PBS -p +4
#PBS -N HelloWorld
#PBS -l storage=gdata/v46+gdata/hh5+gdata/e14+scratch/v46+scratch/v45+scratch/e14
#PBS -W umask=027
#PBS -j n 

##PBS -wd
#PBS -joe


# Test bash script to demonstrate resubmissions
script_name='resub.sh'

#script_name='./diablo > output.dat &'


# Set default values of count and max
if [ -z $count ]; then
    count=1
fi

if [ -z $max ]; then
    max=$count
fi

# Log submission counters
echo "Run $count of $max"
echo "$PBS_O_WORKDIR"
# if [ $count -gt 1 ]; then
#     # Copy previous restart files to input path
#     cd $PBS_O_WORKDIR
#     ./copy_start_saved 
#     echo 'Copy files here'
# fi

# Run the model
#module load openmpi
#module load ipm
#module load totalview/8.7.0-3
#export IPM_LOGDIR=/short/x52/bishakh/ipm_logs
#export IPM_LOGFILE=$PBS_JOBID.$USER.$PROJECT.`date +%s`

julia --project -e '@info "Hello World"'
#mpirun --debug -n 49 ./diablo > output.dat -a

((count++))

if [ $count -le $max ]; then
    echo "Resubmitting model"
    cd $PBS_O_WORKDIR
    qsub -v count=$count,max=$max $script_name
else
    echo "Last submission"
fi
