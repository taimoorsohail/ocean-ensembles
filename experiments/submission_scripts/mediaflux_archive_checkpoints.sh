#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --job-name=mediaflux_ckpt
#SBATCH --output=../run_logs/mediaflux_ckpt_%j.out
#SBATCH --error=../run_logs/mediaflux_ckpt_%j.err
#SBATCH --export=ALL

set -euo pipefail

if [ -f "${HOME}/.bashrc" ]; then
  source "${HOME}/.bashrc"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

cd "${repo_root}"

julia --project="${repo_root}/experiments/submission_scripts" \
  "${repo_root}/experiments/submission_scripts/mediaflux_archive_checkpoints.jl"
