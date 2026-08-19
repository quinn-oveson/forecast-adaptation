#!/bin/bash --login
# SLURM array job for the uniform forcing-shift adaptation experiment (run_shift.py).
#
# WHICH EXPERIMENT THIS RUNS IS NOT IN THIS FILE, and is not read from cfg/ either. Every
# submission names a frozen experiment directory via L96_EXP:
#   sbatch --export=ALL,L96_EXP=results/shift --array=0-4 slurm/shift_array.sbatch
#
# That directory must already contain config.resolved.yaml and grid.yaml, written by:
#   python run_shift.py --config cfg/shift.yaml --seeds 0 1 2 3 4 --out-dir results/shift --freeze
#
# Freezing matters more here than a preset env var would. Each compute node re-reads the config
# when its task actually STARTS, so editing cfg/shift.yaml while an array sits in the queue
# would silently change what those queued tasks run -- and unlike a preset name, a YAML file
# gives no hint in squeue that it changed. The frozen snapshot is also the run's provenance
# record (P2): it is the exact resolved config, not the file that happened to be on disk later.
#
# ---------------------------------------------------------------------------
# ONE ARRAY TASK = ONE CELL = one (arm, seed).
#
# Cells are ordered stage-major, so each stage is a CONTIGUOUS range -- but do not hand-write
# them, because the ranges move when --replay, --seeds or --conv-mult change. Ask:
#   python run_shift.py --exp-dir results/shift --print-array-specs
#   python run_shift.py --exp-dir results/shift --list-cells
#
# For the default grid (5 replay ratios x 5 seeds, conv-mult 10) that is 45 tasks:
#   pretrain  0-4    5 tasks   no dependency   1x budget
#   cold      5-14  10 tasks   no dependency   1x budget    cold_all, cold_new
#   conv     15-19   5 tasks   no dependency  10x budget    cold_all_conv (ceiling only)
#   warm     20-44  25 tasks   AFTER pretrain  1x budget    replay sweep + warm_new
#
# THE WARM STAGE HAS A REAL DEPENDENCY. Every warm task loads
# results/<exp>/checkpoints/pretrain_s<seed>/best_next.pt, which the pretrain stage writes.
# Submitting warm without --dependency=afterok on the pretrain array makes all 25 warm tasks
# die immediately with a missing-checkpoint error. submit_shift_experiment.sh wires this for
# you; if you submit by hand, wire it yourself.
# ---------------------------------------------------------------------------
# GPU SELECTION (passed at submission, deliberately not a #SBATCH line here, because node
# names drift and a stale hardcoded list is worse than none).
#
# Inventory as recorded by the predecessor repo (weight-init-effects, round1):
#   m9g-1-[1-20], m9g-2-[1-20]   gpu:p100:4  = 160 GPUs  (features: public,gpu,pascal)
#   m13h-1-[1-2], m13h-2-[1-2]   gpu:h200:8  =  32 GPUs  (feature: hopper)
#   m13l-1-[1-2], m13l-2-[1-2]   gpu:l40s:4  =  16 GPUs  (feature: lovelace)
#   mgh-1-[1-2]                  gpu:gh200:1 -- ARM, the x86_64 conda env will NOT run there
#
# NOTE: m13h/m13l do NOT carry the `gpu` feature, so --constraint=gpu excludes exactly the
# fast nodes. Use positive selection instead, which fails loudly rather than silently landing
# on Pascal:
#       --constraint='hopper|lovelace'
#
# This job is partly CPU-bound no matter which GPU it gets: each task generates its own L96
# trajectories, 150,000 NumPy RK4 steps measured at ~24 s on an M-series laptop, plus CPU-side
# SVDs for effective rank. m9g nodes are Broadwell-era, so they are slow on exactly the part a
# faster GPU cannot help with. The model is small (2.63M params at hidden=512), so the GPU is
# not the bottleneck anyway -- prefer starting sooner over a bigger GPU if the queue is long.
# ---------------------------------------------------------------------------

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1     # Nothing here is CPU-parallel; 1 core schedules sooner.
#SBATCH --gpus=1
#SBATCH --mem=8G              # PLACEHOLDER. The predecessor measured MaxRSS 1.3-2.1 GB on the
                               # same substrate at comparable sizes, and this experiment's
                               # pooled tensor is only ~14 MB, so 8G is generous rather than
                               # measured. Confirm with sacct after calibration (below).
#SBATCH --time=00:30:00       # PLACEHOLDER for the 1x tiers. The conv tier is 10x the step
                               # budget and needs its own --time -- see --print-array-specs.
                               # DO NOT submit a full array until calibration has replaced
                               # these with measured numbers.
#SBATCH --array=0-4           # PLACEHOLDER -- always override with --array= at submission.
#SBATCH -J l96_shift
#SBATCH --output=slurm_logs/shift_%A_%a.out
#SBATCH --error=slurm_logs/shift_%A_%a.err

# mkdir -p slurm_logs before submitting -- SLURM will not create it for you.

# Fail loudly rather than defaulting to some directory and writing results where nobody
# looks for them.
: "${L96_EXP:?not set -- submit with --export=ALL,L96_EXP=results/shift}"

mamba activate forecast-adaptation   # see ONE-TIME SETUP at the bottom of this file

cd "$SLURM_SUBMIT_DIR"   # the repo root you ran sbatch from (NOT $(dirname "$0") -- SLURM
                          # copies this script to /var/spool/slurmd before running it)

# Without this the .out file stays EMPTY until the task exits, because Python block-buffers
# stdout when it is a file rather than a tty. Verified locally: a 40-minute run had written
# five result rows and zero bytes of log. That makes a running job impossible to monitor and
# makes a timed-out job look like it never started. Neither predecessor repo's sbatch scripts
# set this.
export PYTHONUNBUFFERED=1

# #SBATCH -J is fixed at submission time, before $SLURM_ARRAY_TASK_ID exists, so every task
# would otherwise appear in squeue as an identical row. Rename at runtime by decoding through
# run_shift.decode_task_id() -- the same source of truth the run itself uses -- rather than
# duplicating the grid here where it could drift.
CELL_NAME=$(python -c "
import yaml
from pathlib import Path
from run_shift import decode_task_id, GRID_SNAPSHOT
grid = yaml.safe_load(open(Path('${L96_EXP}') / GRID_SNAPSHOT))
c = decode_task_id(${SLURM_ARRAY_TASK_ID}, grid)
name = c.arm if c.stage == c.arm else f'{c.stage}_{c.arm}'
print(f'{name}_seed{c.seed}')
")
# Address the array ELEMENT as <arrayjobid>_<taskid>; $SLURM_JOB_ID can land the rename on a
# sibling task (observed in the catdd sweep, job 13029191, 2026-08-03).
scontrol update JobId="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" JobName="$CELL_NAME" || true

python run_shift.py --exp-dir "$L96_EXP" --task-id "$SLURM_ARRAY_TASK_ID"

# ============================================================================
# NORMAL PATH -- the whole experiment in one command:
#
#     ./slurm/submit_shift_experiment.sh results/shift
#
# That freezes the config, submits one array per stage with the right --array/--mem/--time,
# gates the warm stage on the pretrain array with --dependency=afterok, and submits the
# aggregation job gated on everything. Preview without submitting: --dry-run.
# Everything below is the manual fallback.
#
# ONE-TIME SETUP (before the first submission)
#   module load miniforge3
#   mamba env create -f environment-cluster.yml     # creates `forecast-adaptation`
#   mamba activate forecast-adaptation
#   mkdir -p slurm_logs results
# No data transfer is needed: L96 trajectories are generated from the equations at runtime, so
# compute nodes need no outbound internet and no dataset copy.
#
# MANUAL ROLLOUT
# 1. Freeze the experiment (also prints the array specs):
#      python run_shift.py --config cfg/shift.yaml --seeds 0 1 2 3 4 \
#             --out-dir results/shift --freeze
#
# 2. Calibrate ONE cell of each tier before committing an array. The 1x and 10x tiers have
#    unrelated cost profiles, so timing one tells you nothing about the other:
#      mkdir -p slurm_logs
#      sbatch --export=ALL,L96_EXP=results/shift --constraint='hopper|lovelace' \
#             --array=0  --time=00:30:00 --mem=8G slurm/shift_array.sbatch   # 1x  (pretrain)
#      sbatch --export=ALL,L96_EXP=results/shift --constraint='hopper|lovelace' \
#             --array=15 --time=04:00:00 --mem=8G slurm/shift_array.sbatch   # 10x (conv)
#    Then read the REAL numbers off the completed jobs:
#      sacct -X -j <jobid> --format=JobID,JobName%30,NodeList,MaxRSS,Elapsed,State
#    and replace the placeholders above with those plus margin.
#
# 3. Submit the stages. pretrain FIRST and on its own, because warm depends on it:
#      pre=$(sbatch --parsable --export=ALL,L96_EXP=results/shift \
#            --constraint='hopper|lovelace' --array=0-4 --time=<t> --mem=<m>G \
#            slurm/shift_array.sbatch)
#      sbatch --export=ALL,L96_EXP=results/shift --constraint='hopper|lovelace' \
#             --array=5-14  --time=<t>  --mem=<m>G slurm/shift_array.sbatch
#      sbatch --export=ALL,L96_EXP=results/shift --constraint='hopper|lovelace' \
#             --array=15-19 --time=<t10> --mem=<m>G slurm/shift_array.sbatch
#      sbatch --export=ALL,L96_EXP=results/shift --constraint='hopper|lovelace' \
#             --array=20-44 --time=<t>  --mem=<m>G --dependency=afterok:$pre \
#             slurm/shift_array.sbatch
#
# 4. Aggregate once everything finishes:
#      python aggregate_shift.py --exp-dir results/shift
#
# RE-RUNNING FAILURES
# Tasks are independent and each writes only its own results/<exp>/tasks/task<NNN>_*.csv, and
# those files are truncated on open, so a failed or PREEMPTED task can simply be resubmitted
# with --array=<id> -- nothing needs deleting first, and automatic requeue is safe.
# aggregate_shift.py names the missing task ids for you and refuses to summarise a partial
# grid unless you pass --allow-incomplete.
# One exception: re-running a pretrain task rewrites the checkpoint that its five warm cells
# started from, so those warm cells must be re-run too if you want the chain to be consistent.
# ============================================================================
