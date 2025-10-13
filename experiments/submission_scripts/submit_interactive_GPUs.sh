mpirun -n 3 julia --project \
  ../../mwes/JLDSave_distributed.jl\
  > ../run_logs/JLDSave_out.stdout \
  2> ../run_logs/JLDSave_out.stderr
