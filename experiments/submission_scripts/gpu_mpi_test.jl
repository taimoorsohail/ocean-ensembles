using MPI
using CUDA

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

# Select a GPU for this rank
CUDA.device!(rank % length(CUDA.devices()))

# Each rank has a GPU array
a_gpu = CUDA.fill(rank+1.0f0, 4)

# Prepare receive buffer only on root
b_gpu = rank == 0 ? CUDA.zeros(Float32, 4*nprocs) : nothing

# Correct CUDA-aware gather
MPI.Gather!(a_gpu, b_gpu, 0, comm)

# Print on root
if rank == 0
    println("Gathered GPU array: ", Array(b_gpu))
end

MPI.Barrier(comm)
MPI.Finalize()
