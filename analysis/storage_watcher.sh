#!/bin/bash
#SBATCH --job-name=storage_watcher
#SBATCH --partition=sapphire
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/gpfs/projects/punim2499/taimoor/gadi-ensembles/postprocessed/_pipeline/slurm-%j.out
#
# Keeps the project disk to: code + inputs, the output files being written, the post-processed set,
# and the trailing checkpoint of each run. One pass, for every run folder outputs/<run>/:
#   1. closed output files -> postprocess_chunk.jl -> Mediaflux <run>/raw/ -> verify -> delete locally
#   2. checkpoints_to_archive/* (hard links made by the run) -> Mediaflux <run>/checkpoints/ -> verify -> delete link
# A file is closed when a later part of the same stream exists, or a later run's file exists in the folder,
# or the pass is run with FINAL=1 (after a run has finished for good).
# Submitted as a Slurm job it resubmits itself every WATCH_INTERVAL; touch postprocessed/_pipeline/STOP to end the chain.
# Usage:  sbatch storage_watcher.sh        (chain)       bash storage_watcher.sh      (one pass, here)
#         FINAL=1 bash storage_watcher.sh  (also flush the last open files of finished runs)

set -u
ROOT=/data/gpfs/projects/punim2499/taimoor/gadi-ensembles
OUT=${OUT_ROOT:-$ROOT/outputs}
PP=${PP_ROOT:-$ROOT/postprocessed}
PIPE=$PP/_pipeline
MF=${MF_ROOT:-/projects/proj-6700_oceanensembles-1128.4.1306/tsohail/RYF_sxtdeg}
ENVDIR=$ROOT/experiments/submission_scripts
WATCH_INTERVAL=${WATCH_INTERVAL:-30minutes}
MIN_FREE_GB=${MIN_FREE_GB:-300}
FINAL=${FINAL:-0}
mkdir -p "$PIPE" "$PIPE/mf_logs"
LOG=$PIPE/watcher.log
MANIFEST=$PIPE/manifest.tsv     # date  run  file  bytes  mediaflux_path
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# Keep the chain alive first, so a failed pass does not end it
if [ "${SLURM_JOB_NAME:-}" = storage_watcher ] && [ ! -e "$PIPE/STOP" ]; then
    if [ "$(squeue -h -u "$USER" -n storage_watcher -t PENDING | wc -l)" -eq 0 ]; then
        sbatch --begin=now+$WATCH_INTERVAL "$ROOT/analysis/storage_watcher.sh" > /dev/null && log "next pass queued in $WATCH_INTERVAL"
    fi
fi

# One pass at a time across nodes (flock does not lock across GPFS clients): mkdir is atomic.
# The lock records its owner (Slurm job id, or pid@host); a lock whose job has ended is stale.
LOCKDIR=$PIPE/lock.d
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    owner=$(cat "$LOCKDIR/owner" 2>/dev/null)
    if [[ "$owner" =~ ^[0-9]+$ ]]; then
        squeue -h -j "$owner" 2>/dev/null | grep -q . && { log "pass $owner still running; skipping"; exit 0; }
    elif [ -n "$(find "$LOCKDIR" -maxdepth 0 -mmin -240)" ]; then
        log "pass $owner still running; skipping"; exit 0
    fi
    log "removing stale lock of $owner"; rm -rf "$LOCKDIR"
    mkdir "$LOCKDIR" 2>/dev/null || { log "lost the lock race; skipping"; exit 0; }
fi
echo "${SLURM_JOB_ID:-$$@$(hostname)}" > "$LOCKDIR/owner"
trap 'rm -rf "$LOCKDIR"' EXIT

module load unimelb-mf-clients Julia/1.12.6 > /dev/null 2>&1

# Size of an asset in Mediaflux (empty if missing)
mf_size() {
    aterm --app exec asset.get :id "path=$1" :xpath -ename size content/size 2>/dev/null | grep -o '"[0-9]*"' | tr -d '"'
}

# Upload files to a Mediaflux collection; echo the ones whose upload verified (checksum at upload + size)
upload_verified() {
    local dest=$1; shift
    unimelb-mf-upload --csum-check --nb-workers 4 --split --create-parents --quiet \
        --log-dir "$PIPE/mf_logs" --dest "$dest" "$@" > "$PIPE/last_upload.txt" 2>&1
    local status=$?
    grep -q "Failed files: *0 files" "$PIPE/last_upload.txt" || status=1
    [ $status -eq 0 ] || { log "upload to $dest failed (exit $status): $(tr -s ' ' < "$PIPE/last_upload.txt" | tail -3 | tr '\n' ' ')"; }
    for f in "$@"; do
        local local_size remote_size
        local_size=$(stat -c %s "$f"); remote_size=$(mf_size "$dest/$(basename "$f")")
        if [ "$local_size" = "$remote_size" ]; then echo "$f"; else log "verify failed for $f (local $local_size, Mediaflux ${remote_size:-missing})"; fi
    done
}

# stream key without the _partN suffix; run number from _runNNNN
stem_of() { basename "$1" .jld2 | sed -E 's/_part[0-9]+$//'; }
part_of() { basename "$1" .jld2 | grep -oE '_part[0-9]+$' | grep -oE '[0-9]+' || echo 0; }
run_of()  { basename "$1" | grep -oE '_run[0-9]+' | grep -oE '[0-9]+' | sed 's/^0*//'; }

for rundir in "$OUT"/*/; do
    run=$(basename "$rundir")
    shopt -s nullglob
    outputs=()
    for f in "$rundir"*.jld2; do [[ $(basename "$f") == *checkpoint* ]] || outputs+=("$f"); done

    # 1. closed output files
    closed=()
    max_run=0
    for f in "${outputs[@]}"; do r=$(run_of "$f"); [ -n "$r" ] && [ "$r" -gt "$max_run" ] && max_run=$r; done
    for f in "${outputs[@]}"; do
        s=$(stem_of "$f"); p=$(part_of "$f"); r=$(run_of "$f")
        newer=0
        for g in "${outputs[@]}"; do
            [ "$(stem_of "$g")" = "$s" ] && [ "$(part_of "$g")" -gt "$p" ] && newer=1
        done
        if [ "$newer" = 1 ] || { [ -n "$r" ] && [ "$r" -lt "$max_run" ]; } || [ "$FINAL" = 1 ]; then
            closed+=("$f")
        fi
    done

    if [ ${#closed[@]} -gt 0 ]; then
        log "$run: ${#closed[@]} closed file(s): $(for f in "${closed[@]}"; do basename "$f"; done | tr '\n' ' ')"
        (cd "$ENVDIR" && julia --project "$ROOT/analysis/postprocess_chunk.jl" "$PP/$run" "${closed[@]}") > "$PIPE/last_postprocess.txt" 2>&1
        processed=$(grep '^POSTPROCESSED ' "$PIPE/last_postprocess.txt" | cut -d' ' -f2-)
        uploaded=$(upload_verified "$MF/$run/raw" "${closed[@]}")
        for f in "${closed[@]}"; do
            if grep -qxF "$f" <<< "$processed" && grep -qxF "$f" <<< "$uploaded"; then
                printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$run" "$(basename "$f")" "$(stat -c %s "$f")" "$MF/$run/raw" >> "$MANIFEST"
                rm -f -- "$f" && log "$run: moved $(basename "$f")"
            else
                grep -qxF "$f" <<< "$processed" || log "$run: post-processing failed for $(basename "$f") (see last_postprocess.txt); kept locally"
            fi
        done
    fi

    # 2. archived checkpoints (hard links, so deleting them frees the space once the run has moved on)
    links=("$rundir"checkpoints_to_archive/*.jld2)
    if [ ${#links[@]} -gt 0 ]; then
        for f in $(upload_verified "$MF/$run/checkpoints" "${links[@]}"); do
            printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$run" "$(basename "$f")" "$(stat -c %s "$f")" "$MF/$run/checkpoints" >> "$MANIFEST"
            rm -f -- "$f" && log "$run: archived checkpoint $(basename "$f")"
        done
    fi
done

free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    log "WARNING: only ${free_gb} GB free on the project disk"
    echo "$(date '+%F %T') only ${free_gb} GB free" > "$PIPE/LOW_DISK"
else
    rm -f "$PIPE/LOW_DISK"
fi
log "pass done, ${free_gb} GB free"
