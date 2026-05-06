#!/usr/bin/env bash
set -euo pipefail

mpiexec --bind-to socket --map-by socket --report-bindings -n 4 \
  julia --project=. ../RYF_3deg_GPU.jl --arch GPU --time 1 \
  2>&1 | tee run_3deg.log
