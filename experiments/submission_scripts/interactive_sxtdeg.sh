#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nranks="${1:-2}"
julia_bin="${JULIA_BIN:-$(command -v julia)}"

if command -v module >/dev/null 2>&1; then
  module purge
fi

export PATH="$(dirname "${julia_bin}"):/usr/local/bin:/usr/bin:/bin"

cd "${script_dir}"

"${script_dir}/mpirun" --oversubscribe --bind-to none -n "${nranks}" \
  "${julia_bin}" --project="${script_dir}" "${script_dir}/../RYF_sxtdeg.jl" --arch GPU --time 1 \
  2>&1 | tee run_sxtdeg.log
