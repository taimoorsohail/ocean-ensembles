import numpy as np
import subprocess, os

list_GPUs = [1,4,8]
list_GPUs_submission = [1, 4, 8]

list_CPUs = [int(2*12), int(3*12), int(4*12), 96, 96, 96, 96]

for i in range(len(list_GPUs)):
    print('Submitting job for '+str(list_GPUs[i])+ 'GPUs')
    file = open("submit_"+str(list_GPUs[i])+"_GPUs.sh", "w")
    file.write("#!/bin/bash\n")

    file.write('#PBS -P ui41\n')
    file.write('#PBS -q gpuvolta\n')
    file.write('#PBS -l walltime=12:00:00\n')
    file.write('#PBS -l mem=150GB\n')
    file.write('#PBS -l storage=gdata/v46+gdata/hh5+gdata/e14+scratch/v46+scratch/v45+scratch/e14\n')
    file.write('#PBS -l wd\n')
    file.write('#PBS -l ncpus='+str(list_CPUs[i])+'\n')
    file.write('#PBS -l ngpus='+str(list_GPUs_submission[i])+'\n')
    file.write('#PBS -l jobfs=180GB\n')
    file.write('#PBS -W umask=027\n')
    file.write('#PBS -j n \n')
    file.write('#PBS -N '+str(list_GPUs[i])+'_GPUs\n')

    file.write('# Output logs\n')
    file.write('#PBS -o profile_scalings_'+str(list_GPUs[i])+'GPUs.o\n')
    file.write('#PBS -e profile_scalings_'+str(list_GPUs[i])+'GPUs.e\n')

    #cmd = f"""mpiexec -np {list_GPUs[i]} bash -c '
    #nsys profile \
    #    --trace=cuda,mpi \
    #    --force-overwrite true \
    #    --output=my_profile${{OMPI_COMM_WORLD_RANK}} \
    #    julia --project --check-bounds=no scaling_ClimaOcean.jl \
    #    > profile_scalings_{list_GPUs[i]}GPUs.stdout \
    #    2> profile_scalings_{list_GPUs[i]}GPUs.stderr
    #' """
    cmd = f"""mpiexec -np {list_GPUs[i]} bash -c '
        julia --project --check-bounds=no scaling_ClimaOcean.jl \
        > profile_scalings_{list_GPUs[i]}GPUs.stdout \
        2> profile_scalings_{list_GPUs[i]}GPUs.stderr
    ' """
    file.write(cmd)
    file.close()

    jobfile = f"submit_{list_GPUs[i]}_GPUs.sh"
    subprocess.run(['qsub', jobfile])
    #os.remove(jobfile)
