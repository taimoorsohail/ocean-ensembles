using OceanEnsembles

total_ranks = MPI.Comm_size(MPI.COMM_WORLD)

ranks = 0:total_ranks
prefix = output_path * "fluxes_sxthdeg_iteration0"
prefix_out = output_path * "fluxes_sxthdeg_iteration0"

