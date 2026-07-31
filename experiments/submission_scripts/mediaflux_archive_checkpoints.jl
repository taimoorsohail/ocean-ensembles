const DEFAULT_CHECKPOINT_PREFIX = "RYF_sxtdeg_checkpoint"
const DEFAULT_OUTPUT_PATH = normpath(joinpath(@__DIR__, "..", "..", "outputs"))
const DEFAULT_MEDIAFLUX_CONFIG = expanduser("~/.Arcitecta/mflux.cfg")
const DEFAULT_MEDIAFLUX_DEST_SUFFIX = ("ocean-ensembles", "outputs")
const DEFAULT_MEDIAFLUX_UPLOAD_CMD = "unimelb-mf-upload"
const DEFAULT_MEDIAFLUX_WORKERS = 4
const MANIFEST_NAME = ".mediaflux_archived_checkpoints.log"
const LOCKDIR_NAME = ".mediaflux_archive.lock"

required_env(name) = get(ENV, name, "") == "" ? throw(ArgumentError("Environment variable $name must be set")) : ENV[name]

function checkpoint_iteration(filename)
    match_data = match(r"_iteration(\d+)\.jld2$", basename(filename))
    isnothing(match_data) && return nothing
    return parse(Int, match_data.captures[1])
end

function checkpoint_files(output_dir, checkpoint_prefix)
    prefix = checkpoint_prefix * "_iteration"
    files = String[]

    for name in readdir(output_dir)
        path = joinpath(output_dir, name)
        if !isfile(path)
            continue
        elseif !startswith(name, prefix) || !endswith(name, ".jld2")
            continue
        elseif isnothing(checkpoint_iteration(name))
            continue
        end

        push!(files, path)
    end

    sort!(files; by=path -> checkpoint_iteration(path))
    return files
end

function read_manifest(manifest_path)
    if !isfile(manifest_path)
        return Set{String}()
    end

    archived = Set{String}()
    for line in eachline(manifest_path)
        entry = strip(line)
        isempty(entry) || push!(archived, entry)
    end

    return archived
end

function append_manifest!(manifest_path, files)
    isempty(files) && return nothing
    open(manifest_path, "a") do io
        for file in files
            println(io, basename(file))
        end
    end
    return nothing
end

function join_mediaflux_path(parts::AbstractString...)
    cleaned = String[]
    for part in parts
        stripped = strip(part, '/')
        isempty(stripped) || push!(cleaned, stripped)
    end
    return join(cleaned, "/")
end

function upload_files!(files; mf_config, remote_project_path)
    isempty(files) && return nothing

    upload_cmd = get(ENV, "MEDIAFLUX_UPLOAD_CMD", DEFAULT_MEDIAFLUX_UPLOAD_CMD)
    nb_workers = parse(Int, get(ENV, "MEDIAFLUX_NB_WORKERS", string(DEFAULT_MEDIAFLUX_WORKERS)))
    destination = join_mediaflux_path(remote_project_path, DEFAULT_MEDIAFLUX_DEST_SUFFIX...)

    cmd = Cmd([
        upload_cmd,
        "--mf.config", mf_config,
        "--dest", destination,
        "--create-parents",
        "--nb-workers", string(nb_workers),
        "--preserve-modified-time",
        files...,
    ])

    @info "Uploading checkpoint files to Mediaflux" destination count=length(files) nb_workers
    run(cmd)
    return nothing
end

function delete_files!(files)
    isempty(files) && return nothing

    for file in files
        if isfile(file)
            @info "Deleting archived checkpoint file" file
            rm(file; force=true)
        end
    end

    return nothing
end

function main()
    remote_project_path = required_env("MEDIAFLUX_PROJ_PATH")
    mf_config = expanduser(get(ENV, "MEDIAFLUX_MF_CONFIG", DEFAULT_MEDIAFLUX_CONFIG))
    output_dir = expanduser(get(ENV, "MEDIAFLUX_ARCHIVE_OUTPUT_PATH", DEFAULT_OUTPUT_PATH))
    checkpoint_prefix = get(ENV, "MEDIAFLUX_ARCHIVE_CHECKPOINT_PREFIX", DEFAULT_CHECKPOINT_PREFIX)

    isdir(output_dir) || throw(ArgumentError("Output directory does not exist: $output_dir"))
    isfile(mf_config) || throw(ArgumentError("Mediaflux config does not exist: $mf_config"))

    manifest_path = joinpath(output_dir, MANIFEST_NAME)
    lockdir_path = joinpath(output_dir, LOCKDIR_NAME)

    try
        mkdir(lockdir_path)
    catch err
        if err isa Base.IOError || err isa SystemError
            @warn "Another Mediaflux archive job appears to be active; skipping this run" lockdir_path
            return nothing
        end
        rethrow(err)
    end

    try
        files = checkpoint_files(output_dir, checkpoint_prefix)
        if length(files) <= 1
            @info "Nothing to archive yet; keeping the most recent checkpoint locally" count=length(files)
            return nothing
        end

        archived = read_manifest(manifest_path)
        latest_file = last(files)
        older_files = files[1:end-1]

        already_uploaded = [file for file in older_files if basename(file) in archived]
        pending_upload = [file for file in older_files if !(basename(file) in archived)]

        if isempty(already_uploaded) && isempty(pending_upload)
            @info "No checkpoint files are eligible for archiving"
            return nothing
        end

        if !isempty(pending_upload)
            upload_files!(pending_upload;
                          mf_config,
                          remote_project_path)
            append_manifest!(manifest_path, pending_upload)
        end

        deletable_files = vcat(already_uploaded, pending_upload)
        delete_files!(deletable_files)

        @info "Mediaflux checkpoint archive completed" latest_checkpoint=basename(latest_file) uploaded=length(pending_upload) deleted=length(deletable_files)
    finally
        isdir(lockdir_path) && rm(lockdir_path; force=true, recursive=true)
    end

    return nothing
end

main()
