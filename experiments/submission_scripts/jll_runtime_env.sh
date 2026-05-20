# Source this before running the JLL-based Julia/MPI/CUDA setup.
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_NUM_THREADS=1
export JULIA_PKG_PRECOMPILE_AUTO=0

# Let OpenMPI_jll choose a GPU-capable transport. Forcing TCP here can make MPI
# try to send GPU device pointers with writev(2), which fails with Bad address.
unset OMPI_MCA_pml
unset OMPI_MCA_btl

export OPENSSL_ARTIFACT=/data/projects/punim2499/taimoor/.julia/artifacts/5aa05123e9ebbfaf1da781168fed89c31b8b7497/lib
export LD_LIBRARY_PATH="$OPENSSL_ARTIFACT"
export LD_PRELOAD="$OPENSSL_ARTIFACT/libcrypto.so.3"

unset OPAL_PREFIX
unset OMPI_MCA_plm_slurm_args
unset OMPI_MCA_btl_tcp_if_include
unset OMPI_MCA_btl_openib_if_include
unset OMPI_MCA_oob_tcp_if_include
unset PMIX_MCA_gds
