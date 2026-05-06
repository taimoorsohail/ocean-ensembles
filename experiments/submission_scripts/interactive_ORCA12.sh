#!/usr/bin/env bash
set -euo pipefail


mpiexec --bind-to socket --map-by socket -n 4 \
  julia --project ../ORCA12_expt.jl --arch GPU --time 1 \
  2>&1 | tee run_ORCA12.log
