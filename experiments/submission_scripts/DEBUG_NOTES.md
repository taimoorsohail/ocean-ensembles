# Debugging notes (running log)

Spartan wipes agent memory (`~/.claude`) every 24 h. This file is the persistent
record of debugging context. Read it at the start of each session; append to it
as things change. Newest entries go at the top of the History section.

## Persistent tmux setup

- Start a hold job: `gpu-interactive-hold` Slurm job; its output file
  (`~/gpu-interactive-hold-<jobid>.out`) prints the node and socket.
- Socket: `~/.tmux-sockets/job-<jobid>.sock`, session name `work`
  (see `_jobtmux_attach` / `worktmux` in `~/.bashrc`).
- Attach: `sj <jobid>` (alias for `sshjob`, `~/.bashrc:171`).
- Pane logs: `submission_scripts/logs/worktmux_job_<jobid>.log`
  (client log: `worktmux_client_job_<jobid>.log`).
- The tmux server runs on the compute node; from the login node
  `tmux -S <socket> ...` reports "no server running". That does not mean it is dead.
- Stale sockets from July: `~/.tmux-sockets/worktmux_job_27599856*.sock`.

## Current run

- Command (`~/.tsohail_cmd.txt`):
  ```julia
  state = nothing; reclaim_gpu_memory!(); include("../RYF_sxtdeg_serial.jl")
  state = build_simulation(GPU(), 1; add_outputs=true, avid=false)
  run_segment!(state; pickup=false, reconcile_pickup=false, Δt=14minutes, stop_time=10000days)
  ```
- Script: `experiments/RYF_sxtdeg_serial.jl` (`run_segment!` at ~line 1064).

## Open issue: DomainError on first time steps (GPU, H100)

- Seen in job 32846302 (pane log, ~line 129), Δt = 14 min, pickup = false.
- Initialization and the initial time step complete, then:
  `DomainError ... This operation requires a complex input to return a complex result`
  (a sqrt/log/^ of a negative number inside a GPU kernel), raised from
  `update_state!` -> `update_model_field_time_series!` in Oceananigans.
- Initial extrema looked sane: T (-1.89, 31.25) °C, S (23.69, 40.64) g/kg.
- To get the kernel source location: run Julia with `-g2`.
- Related earlier Codex thread: "Debug negative salinities in SJ" (2026-09-14).

## History

- 2026-10-26: Job 33698681 (`spartan-gpgpu180`, started 2026-10-24 07:24, 7-day
  limit). User ran the current command manually in tmux via `sj 33698681`.
  No code changes requested from the agent. Outcome of this run not yet recorded.
- 2026-10-26: Previous notes file could not be found (likely wiped or never
  saved somewhere persistent); this file was rebuilt from pane logs.
- 2026-10-19: Job 32846302: last attach 14:08; DomainError above reproduced
  with the current command.
- 2026-10-14: Job 32263557 pane log ends (not yet reviewed).
