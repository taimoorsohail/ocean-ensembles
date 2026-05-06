#!/usr/bin/env bash
set -euo pipefail

mpiexec --bind-to socket --map-by socket --report-bindings -n 4 \
  julia --project=. ../RYF_sxtdeg.jl --arch GPU --time 1 \
  2>&1 | tee run_sxtdeg.log
