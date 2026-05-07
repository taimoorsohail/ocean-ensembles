#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/.." && pwd)"
project_dir="${COMBINE_PROJECT:-${repo_dir}/experiments/submission_scripts}"
workers="${COMBINE_WORKERS:-2}"
logs_dir="${COMBINE_LOG_DIR:-${project_dir}/logs}"
julia_bin="${JULIA:-julia}"

mkdir -p "${logs_dir}"

echo "Combining rank-split files"
echo "  script:  ${script_dir}/combine2d_distributed.jl"
echo "  project: ${project_dir}"
echo "  workers: ${workers}"
echo "  gc every: ${COMBINE_GC_EVERY:-5} frame(s)"
echo "  logs:    ${logs_dir}/combine2d_worker_*.log"
echo ""

pids=()
for worker in $(seq 1 "${workers}"); do
    log_file="${logs_dir}/combine2d_worker_${worker}.log"
    echo "Starting worker ${worker}/${workers}: ${log_file}"

    COMBINE_WORKERS="${workers}" \
    COMBINE_WORKER="${worker}" \
    COMBINE_GC_EVERY="${COMBINE_GC_EVERY:-5}" \
    stdbuf -oL -eL "${julia_bin}" --project="${project_dir}" \
        -e "include(\"${script_dir}/combine2d_distributed.jl\")" \
        > "${log_file}" 2>&1 &

    pids+=("$!")
done

echo ""
echo "Worker PIDs: ${pids[*]}"
echo "Follow progress with:"
echo "  tail -f ${logs_dir}/combine2d_worker_*.log"
echo ""

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [[ "${status}" -eq 0 ]]; then
    echo "All combine workers finished successfully."
else
    echo "One or more combine workers failed. Check ${logs_dir}/combine2d_worker_*.log" >&2
fi

exit "${status}"
