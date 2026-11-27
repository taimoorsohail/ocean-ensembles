#!/bin/bash
#SBATCH -p gpu-h100
#SBATCH --time=7-00:00:00
#SBATCH --time-min=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --job-name=RYF_control
#SBATCH --output=/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/outputs/RYF_control/slurm-%j.out
#
# One segment of the 100-year RYF control. It picks up the latest checkpoint in outputs/RYF_control/
# and runs until 20 min before this job's time limit, writing a checkpoint at the end.
#
# How segments follow each other:
#  - At the start, each segment submits the next one as an ordinary job with no dependency, so the
#    next one gets its place in the queue without waiting for this one to finish. It is held back
#    (--begin) until QUEUE_WAIT_HOURS before this job ends. QUEUE_WAIT_HOURS is the typical
#    (median) wait for 1-GPU, multi-day gpu-h100 jobs, so the next segment should usually start
#    about when this one ends.
#  - If the next segment starts early, it waits (and does nothing) until the previous segment has
#    left the queue, so two segments never write the same files at once.
#  - If a maintenance window would cut into the next segment, its time limit is shortened so it
#    ends before the window (see segment_minutes below). --time-min also lets Slurm shorten a
#    segment itself, though on Spartan that has not been seen to work. The run reads its real end
#    time, so a shorter job is fine.
#
# The chain stops when the run reaches 100 years, when a segment fails (needs a look), or if
# outputs/RYF_control/STOP_CHAIN exists. (The file STOP was written when the first 60 years finished;
# it is kept as a marker and no longer stops anything.)
set -u
ROOT=/data/gpfs/projects/punim2499/taimoor/gadi-ensembles
OUT=$ROOT/outputs/RYF_control
SCRIPT=$ROOT/experiments/submission_scripts/RYF_control_chain.sh
STOPFILE=$OUT/STOP_CHAIN
QUEUE_WAIT_HOURS=67   # median wait of 1-GPU, 2-7 day gpu-h100 jobs, 1-24 Nov 2026

# Wait for the previous segment to finish, if it is still running.
PREV_JOB=${PREV_JOB:-}
if [ -n "$PREV_JOB" ]; then
    while squeue -h -j "$PREV_JOB" 2>/dev/null | grep -q .; do
        echo "$(date): previous segment $PREV_JOB still in the queue; waiting"
        sleep 60
    done
fi
[ -e $STOPFILE ] && { echo "$STOPFILE present; not running"; exit 0; }

# Time limit (in minutes) for a segment expected to start at time $1 (seconds since 1970).
# Normally 7 days. If a maintenance window begins less than 7 days after that start, the limit is
# cut so the job ends 1 hour before the window, because Slurm will not start a job that would run
# into it. If that leaves less than 1 day, the full 7 days is kept and the job runs after the window.
segment_minutes() {
    local start_s=$1 limit_s=$(( 7 * 86400 )) t gap
    while read -r t; do
        gap=$(( $(date -d "$t" +%s) - start_s - 3600 ))
        (( gap >= 86400 && gap < limit_s )) && limit_s=$gap
    done < <(scontrol show reservation -o 2>/dev/null | grep -E 'Flags=[^ ]*MAINT' | grep -oP 'StartTime=\K\S+')
    echo $(( limit_s / 60 ))
}

end_s=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)
begin_s=$(( end_s - QUEUE_WAIT_HOURS * 3600 ))
(( begin_s < $(date +%s) )) && begin_s=$(date +%s)
next_minutes=$(segment_minutes $end_s)   # the next segment should start about when this one ends
next=$(sbatch --parsable --begin=$(date -d @$begin_s +%Y-%m-%dT%H:%M:%S) --time=$next_minutes \
              --export=ALL,PREV_JOB=$SLURM_JOB_ID $SCRIPT)
echo "next segment queued: $next (eligible from $(date -d @$begin_s), time limit $(( next_minutes / 60 )) h)"

run_id=$(( $(cat $OUT/last_run_id) + 1 ))
echo $run_id > $OUT/last_run_id
echo "run $run_id on $SLURM_NODELIST, job ends $(date -d @$end_s)"

module load Julia/1.12.6
cd $ROOT/experiments/submission_scripts
export RUN_NAME=RYF_control JOB_END=$end_s
julia --project -e '
include("/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/experiments/RYF_sxtdeg_serial.jl")
state = build_simulation(GPU(), parse(Int, ARGS[1]); add_outputs=true)
run_segment!(state; pickup=true, Δt=45minutes, stop_time=100*365days,
             wall_time_limit=parse(Int, ENV["JOB_END"]) - time() - 20*60)
state.simulation.model.clock.time >= 100*365days && touch(joinpath(output_path, "STOP_CHAIN"))
' $run_id
status=$?
if [ $status -ne 0 ]; then
    echo "segment failed (exit $status); stopping the chain"
    touch $STOPFILE
fi
[ -e $STOPFILE ] && scancel $next
