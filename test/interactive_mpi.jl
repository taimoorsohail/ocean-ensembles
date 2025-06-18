using MPI, MPIClusterManagers, Distributed

# Add 3 MPI workers
addprocs(MPIManager(np=3))