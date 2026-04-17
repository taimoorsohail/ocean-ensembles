using MPI
using CUDA
using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Oceananigans.ImmersedBoundaries: GridFittedBottom
using ClimaSeaIce
using ClimaSeaIce.SeaIceDynamics: SeaIceMomentumEquation, SplitExplicitSolver, SemiImplicitStress, ElastoViscoPlasticRheology
using NumericalEarth

MPI.Init()
atexit(MPI.Finalize)

function build_arch(arch_name::String)
    A = uppercase(arch_name)

    if A == "GPU"
        # Oceananigans GPU now requires an explicit backend device.
        return Distributed(GPU(CUDA.CUDABackend());
                           partition = Partition(y = DistributedComputations.Equal()),
                           synchronized_communication = true)
    elseif A == "CPU"
        return Distributed(CPU();
                           partition = Partition(y = DistributedComputations.Equal()),
                           synchronized_communication = true)
    else
        error("Unknown architecture . Use GPU or CPU.")
    end
end

function main()
    arch_name = isempty(ARGS) ? "GPU" : ARGS[1]
    arch = build_arch(arch_name)

    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    Nx, Ny, Nz = 360, 180, 10
    halo = (7, 7, 7)

    @info "MWE: building distributed tripolar grid" rank Nx Ny Nz halo arch_name
    underlying = TripolarGrid(arch; size = (Nx, Ny, Nz), z = (-5000, 0), halo)

    # Keep a simple immersed bottom so the same immersed-advection path is exercised.
    bottom_height(x, y) = -4500.0
    grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom_height); active_cells_map = true)

    dynamics = SeaIceMomentumEquation(grid;
                                      top_momentum_stress = (u = 0.01, v = 0.01),
                                      bottom_momentum_stress = SemiImplicitStress(),
                                      rheology = ElastoViscoPlasticRheology(),
                                      solver = SplitExplicitSolver(grid; substeps = 120))

    model = SeaIceModel(grid;
                        dynamics,
                        ice_thermodynamics = nothing,
                        advection = WENO(order = 7))

    model = sea_ice_simulation(grid, ocean; advection=WENO(order=7)) 


    set!(model; h = 0.2, ℵ = 0.8)

    @info "MWE: running one timestep" rank
    time_step!(model, 10minutes)

    MPI.Barrier(MPI.COMM_WORLD)
    rank == 0 && @info "MWE finished without crash"

    return nothing
end

main()
