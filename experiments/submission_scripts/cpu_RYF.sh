#!/usr/bin/env bash
#SBATCH --job-name=cpu_test
#SBATCH --partition=sapphire
#SBATCH --ntasks=1
#SBATCH --mem=128GB
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --output=/home/tsohail/uom/ocean-ensembles/experiments/run_logs/RYF_sxtdeg_run0001_%j.out
#SBATCH --error=/home/tsohail/uom/ocean-ensembles/experiments/run_logs/RYF_sxtdeg_run0001_%j.err
#SBATCH --export=ALL

set -uo pipefail

export JULIA_CUDA_MEMORY_POOL=none
export JULIA_NUM_THREADS=1

work_root=/data/gpfs/projects/punim2499/taimoor
project=ocean-ensembles/experiments/submission_scripts
simulation=ocean-ensembles/experiments/RYF_sxtdeg_serial.jl
run_id="${RUN_ID:-1}"

if [[ ! "${run_id}" =~ ^[0-9]+$ ]]; then
    echo "RUN_ID must be a non-negative integer; got: ${run_id}" >&2
    exit 2
fi
script_path="$(readlink -f "${BASH_SOURCE[0]}")"

cd "${work_root}"

# Oceananigans stops run! cleanly after 21 hours. The remaining three hours
# cover model construction, the end-of-run checkpoint, and Julia shutdown.
julia --project="${project}" -e "
include(\"${simulation}\")

target_time = Inf
target_iteration = Inf
state = build_simulation(CPU(), ${run_id}; add_outputs=true, Δt=14minutes)
state.simulation.stop_iteration = target_iteration

run_segment!(state;
             pickup=true,
             Δt=14minutes,
             stop_time=target_time,
             wall_time_limit=165hours)

@info \"Batch segment reached its wall-time limit cleanly\" model_time=prettytime(state.simulation)

# Every clean segment requests a continuation; only failures stop the chain.
exit(42)
" 2>&1
status=$?

case "${status}" in
    0)
        echo "Unexpected zero exit from perpetual run; submission chain is complete."
        ;;
    42)
        next_run_id=$((run_id + 1))
        printf -v next_run_label "%04d" "${next_run_id}"
        echo "Final checkpoint completed; submitting continuation as run ${next_run_label}."
        sbatch --export=ALL,RUN_ID="${next_run_id}" \
               --output="/home/tsohail/uom/ocean-ensembles/experiments/run_logs/RYF_sxtdeg_run${next_run_label}_%j.out" \
               --error="/home/tsohail/uom/ocean-ensembles/experiments/run_logs/RYF_sxtdeg_run${next_run_label}_%j.err" \
               "${script_path}"
        ;;
    *)
        echo "Julia failed with status ${status}; not resubmitting." >&2
        exit "${status}"
        ;;
esac
