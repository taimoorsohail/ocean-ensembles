#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

julia_bin="${JULIA_BIN:-$(command -v julia)}"

# This workflow is for the Julia/OpenMPI_jll stack. Keep Spartan MPI modules out
# of LD_LIBRARY_PATH, otherwise OpenMPI_jll can accidentally pick up system PMIx.
if command -v module >/dev/null 2>&1; then
  module purge
fi

export PATH="$(dirname "${julia_bin}"):/usr/local/bin:/usr/bin:/bin"
unset LD_LIBRARY_PATH
unset LIBRARY_PATH
unset CPATH
unset CMAKE_PREFIX_PATH

if [ -f "${script_dir}/jll_runtime_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${script_dir}/jll_runtime_env.sh"
fi

export JULIA_CUDA_MEMORY_POOL="${JULIA_CUDA_MEMORY_POOL:-none}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-1}"
export RYF_SYSIMAGE_ARCH="${RYF_SYSIMAGE_ARCH:-GPU}"
export RYF_SYSIMAGE_PATH="${RYF_SYSIMAGE_PATH:-${script_dir}/RYF_sxtdeg_${RYF_SYSIMAGE_ARCH}.so}"

requested_nranks="${1:-1}"
build_nranks=1
log_dir="${RYF_SYSIMAGE_LOG_DIR:-${script_dir}/logs}"
mkdir -p "${log_dir}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${RYF_SYSIMAGE_LOG:-${log_dir}/build_RYF_sxtdeg_sysimage_${timestamp}.log}"

{
  echo "===== RYF_sxtdeg sysimage build $(date) ====="
  echo "script_dir=${script_dir}"
  echo "julia_bin=${julia_bin}"
  echo "requested_nranks=${requested_nranks}"
  echo "build_nranks=${build_nranks}"
  echo "RYF_SYSIMAGE_ARCH=${RYF_SYSIMAGE_ARCH}"
  echo "RYF_SYSIMAGE_PATH=${RYF_SYSIMAGE_PATH}"
  echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"
  echo "JULIA_CUDA_MEMORY_POOL=${JULIA_CUDA_MEMORY_POOL}"
  echo "PATH=${PATH}"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  if [ "${requested_nranks}" != "${build_nranks}" ]; then
    echo "Note: PackageCompiler creates one sysimage artifact, so the build itself runs on 1 MPI rank."
    echo "      Use the resulting sysimage with ${requested_nranks} ranks in the production run."
  fi
  echo
  echo "Command:"
  printf '  %q' "${script_dir}/mpirun" --oversubscribe --bind-to none -n "${build_nranks}" \
    "${julia_bin}" --project="${script_dir}" "${script_dir}/sysimage/build_RYF_sxtdeg_sysimage.jl"
  echo
  echo
} | tee "${log_file}"

"${script_dir}/mpirun" --oversubscribe --bind-to none -n "${build_nranks}" \
  "${julia_bin}" --project="${script_dir}" "${script_dir}/sysimage/build_RYF_sxtdeg_sysimage.jl" \
  2>&1 | tee -a "${log_file}"
status="${PIPESTATUS[0]}"

echo "Sysimage build log: ${log_file}" | tee -a "${log_file}"

if [ "${status}" -ne 0 ]; then
  echo "Sysimage build failed with exit code ${status}" | tee -a "${log_file}"
  exit "${status}"
fi

echo "Use with: ${julia_bin} -J${RYF_SYSIMAGE_PATH} --project=${script_dir} ${script_dir}/../RYF_sxtdeg.jl --arch ${RYF_SYSIMAGE_ARCH} --time 1"
