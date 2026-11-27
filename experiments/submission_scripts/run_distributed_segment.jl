# One segment of a distributed RYF run, launched on every MPI rank. Set from the environment:
# CELLS_PER_DEGREE, RUN_NAME, OUTPUT_ROOT (read by RYF_sxtdeg_serial.jl), CHILD_ARCH (CPU or GPU),
# DT_MINUTES, STOP_DAYS and JOB_END (Unix time the job ends; a checkpoint is written 20 min before).
# Optional: PICKUP=true restarts from the latest checkpoint; RUN_ID numbers this segment's output files (default 1).
using MPI
MPI.Init()
include("/data/gpfs/projects/punim2499/taimoor/ocean-ensembles/experiments/RYF_sxtdeg_serial.jl")

child_arch = get(ENV, "CHILD_ARCH", "GPU") == "GPU" ? GPU() : CPU()
arch = Distributed(child_arch; partition=Partition(2, 2))

run_id = parse(Int, get(ENV, "RUN_ID", "1"))
state = build_simulation(arch, run_id; add_outputs=true)
run_segment!(state; pickup=get(ENV, "PICKUP", "false") == "true",
             Δt=parse(Float64, ENV["DT_MINUTES"]) * minutes,
             stop_time=parse(Float64, ENV["STOP_DAYS"]) * days,
             wall_time_limit=parse(Int, ENV["JOB_END"]) - time() - 20 * 60)
