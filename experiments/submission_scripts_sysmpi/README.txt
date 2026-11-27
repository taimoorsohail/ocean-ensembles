Copy of submission_scripts' Julia environment (2026-11-27) that uses Spartan's CUDA-aware OpenMPI instead of
OpenMPI_jll (which is built without CUDA support and segfaults when MPI is handed GPU arrays).
Load first, one at a time: GCC/13.3.0, OpenMPI/5.0.3, UCX-CUDA/1.16.0-CUDA-12.6.0, Julia/1.12.6.
Used by claude-notes/scripts/gpu4_hold_runner.sh for the 1/12 run. submission_scripts itself is unchanged.
