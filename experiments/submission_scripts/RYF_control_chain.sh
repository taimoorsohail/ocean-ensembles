#!/bin/bash
#SBATCH -p gpu-h100
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RYF_control
#SBATCH --output=/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/outputs/RYF_control/slurm-%j.out
#
# One segment of the 100-year RYF control: picks up the latest checkpoint in outputs/RYF_control/,
# runs until 20 min before this job's time limit (checkpoint at end), then the next queued job continues.
# The next job is queued at the start (afterany), so it is already waiting when this one ends.
# The chain stops when the run reaches 100 years, when a segment fails (needs a look), or if outputs/RYF_control/STOP exists.
set -u
ROOT=/data/gpfs/projects/punim2499/taimoor/gadi-ensembles
OUT=$ROOT/outputs/RYF_control
SCRIPT=$ROOT/experiments/submission_scripts/RYF_control_chain.sh
[ -e $OUT/STOP ] && { echo "STOP file present; not running"; exit 0; }

next=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID $SCRIPT)
echo "next segment queued: $next"

run_id=$(( $(cat $OUT/last_run_id) + 1 ))
echo $run_id > $OUT/last_run_id
end_s=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)
echo "run $run_id on $SLURM_NODELIST, job ends $(date -d @$end_s)"

module load Julia/1.12.6
cd $ROOT/experiments/submission_scripts
export RUN_NAME=RYF_control JOB_END=$end_s
julia --project -e '
include("/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/experiments/RYF_sxtdeg_serial.jl")
state = build_simulation(GPU(), parse(Int, ARGS[1]); add_outputs=true)
run_segment!(state; pickup=true, Δt=45minutes, stop_time=100*365days,
             wall_time_limit=parse(Int, ENV["JOB_END"]) - time() - 20*60)
state.simulation.model.clock.time >= 100*365days && touch(joinpath(output_path, "STOP"))
' $run_id
status=$?
if [ $status -ne 0 ]; then
    echo "segment failed (exit $status); stopping the chain"
    touch $OUT/STOP
fi
[ -e $OUT/STOP ] && scancel $next
