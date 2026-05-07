using PackageCompiler

project_dir = normpath(joinpath(@__DIR__, ".."))
workload = joinpath(@__DIR__, "precompile_RYF_sxtdeg_workload.jl")
default_sysimage = joinpath(project_dir, "RYF_sxtdeg_$(get(ENV, "RYF_SYSIMAGE_ARCH", "GPU")).so")
sysimage_path = get(ENV, "RYF_SYSIMAGE_PATH", default_sysimage)

packages = [
    :MPI,
    :CUDA,
    :NumericalEarth,
    :ClimaSeaIce,
    :Oceananigans,
    :OceanEnsembles,
    :CFTime,
    :Glob,
    :JLD2,
]

@info "Building RYF_sxtdeg sysimage" project_dir workload sysimage_path packages

create_sysimage(packages;
    project = project_dir,
    sysimage_path,
    precompile_execution_file = workload,
    incremental = true,
    cpu_target = get(ENV, "JULIA_CPU_TARGET", "generic"))

@info "Finished RYF_sxtdeg sysimage" sysimage_path
