#!/bin/bash
# One command to run the whole forcing-shift adaptation experiment on the cluster.
#
#   ./slurm/submit_shift_experiment.sh results/shift              # freeze + submit everything
#   ./slurm/submit_shift_experiment.sh results/shift --dry-run    # print, submit nothing
#
# Submits one array per stage, gates the warm stage on the pretrain array (warm cells load
# pretrain's checkpoints), and gates aggregation on all of them.
#
#   1. pretrain   no dependency          the pre-shift model, one per seed
#   2. cold       no dependency, --nice  cold_all + cold_new
#   3. conv       no dependency, --nice  cold_all_conv, 10x budget, its own walltime tier
#   4. warm       afterok:pretrain       the replay sweep + warm_new
#   5. finalize   afterok:1,2,3,4        aggregate + report
#
# The --array/--mem/--time for each come from run_shift.py --print-array-specs, which derives
# them from the frozen grid. They are NOT written here: the ranges move when --replay, --seeds
# or --conv-mult change, and a hand-written range would silently run the wrong cells.
#
# Optional environment variables:
#   L96_CONFIG      config to freeze from            (default cfg/shift.yaml)
#   L96_SEEDS       seeds to run                     (default "0 1 2 3 4")
#   L96_CONSTRAINT  node constraint                  (default 'hopper|lovelace')
#                   Steers off the 160 Pascal P100s. m13h/m13l do NOT carry the `gpu` feature,
#                   so --constraint=gpu would exclude exactly the fast nodes. Set to empty to
#                   take anything schedulable: L96_CONSTRAINT= ./slurm/submit_shift_experiment.sh ...
#   EXCLUDE_NODES   passed through as --exclude=     (default empty, which is the right default)

set -euo pipefail

EXP="${1:-}"
DRY_RUN=false
case "${2:-}" in
    --dry-run) DRY_RUN=true ;;
    "") ;;
    *) echo "Unknown argument: $2" >&2; exit 2 ;;
esac
if [[ -z "$EXP" ]]; then
    echo "Usage: $0 <exp-dir> [--dry-run]" >&2
    echo "  e.g. $0 results/shift" >&2
    exit 2
fi

CONFIG="${L96_CONFIG:-cfg/shift.yaml}"
SEEDS="${L96_SEEDS:-0 1 2 3 4}"
CONSTRAINT="${L96_CONSTRAINT-hopper|lovelace}"

# A wrapper script is NOT copied to /var/spool/slurmd the way an sbatch script is, so
# $BASH_SOURCE is trustworthy here (unlike inside the .sbatch files). sbatch must be invoked
# from the repo root, because the .sbatch files cd to $SLURM_SUBMIT_DIR.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ARRAY_SBATCH="slurm/shift_array.sbatch"
FINALIZE_SBATCH="slurm/finalize_shift.sbatch"

for f in "$ARRAY_SBATCH" "$FINALIZE_SBATCH" "$CONFIG"; do
    [[ -f "$f" ]] || { echo "ERROR: $REPO/$f not found." >&2; exit 1; }
done

if [[ "$DRY_RUN" == false ]] && ! command -v sbatch > /dev/null; then
    echo "ERROR: sbatch not found -- this has to run on the cluster login node." >&2
    echo "    ssh <netid>@ssh.rc.byu.edu" >&2
    exit 1
fi

# SLURM will not create the log directory and silently fails the job if it is missing.
mkdir -p slurm_logs

# Freeze first. This also runs the stale-results check (it refuses if <exp>/tasks already holds
# results from an earlier run, which aggregation would otherwise mix into the new summary), and
# it is the env check: run_shift.py imports torch, so a missing/unactivated env fails here with
# the real error rather than 45 tasks failing one at a time.
echo "Freezing $EXP from $CONFIG (seeds: $SEEDS)"
if ! freeze_out=$(python run_shift.py --config "$CONFIG" --seeds $SEEDS --out-dir "$EXP" \
                  --freeze 2>&1); then
    echo "ERROR: freeze failed:" >&2
    echo "$freeze_out" | sed 's/^/    /' >&2
    echo "  If that is a ModuleNotFoundError, activate the env first:" >&2
    echo "    module load miniforge3 && mamba activate forecast-adaptation" >&2
    echo "  (module loads do not persist across SSH sessions -- redo them in each new shell)" >&2
    exit 1
fi
echo "$freeze_out" | sed 's/^/  /'

# Re-read the specs from the frozen grid rather than parsing the freeze output, so the
# submission and the tasks agree by construction.
SPECS=$(python run_shift.py --exp-dir "$EXP" --print-array-specs)
[[ -n "$SPECS" ]] || { echo "ERROR: no array specs -- nothing to submit." >&2; exit 1; }

CONSTRAINT_ARG=(); [[ -n "$CONSTRAINT" ]] && CONSTRAINT_ARG=(--constraint="$CONSTRAINT")
EXCLUDE_ARG=();    [[ -n "${EXCLUDE_NODES:-}" ]] && EXCLUDE_ARG=(--exclude="$EXCLUDE_NODES")

echo
printf '%-10s %-10s %6s %10s %7s %s\n' STAGE ARRAY MEM TIME TASKS DEPENDS
while read -r stage spec mem walltime n dep; do
    printf '%-10s %-10s %5sG %10s %7s %s\n' "$stage" "$spec" "$mem" "$walltime" "$n" "$dep"
done <<< "$SPECS"
echo

submit() {  # stage spec mem time nice_flag dependency
    local stage="$1" spec="$2" mem="$3" walltime="$4" nice="$5" dep="$6"
    local cmd=(sbatch --parsable --export=ALL,L96_EXP="$EXP" --array="$spec"
               --mem="${mem}G" --time="$walltime")
    [[ -n "$nice" ]] && cmd+=(--nice)
    [[ -n "$dep" ]] && cmd+=(--dependency=afterok:"$dep")
    cmd+=(${CONSTRAINT_ARG[@]+"${CONSTRAINT_ARG[@]}"} ${EXCLUDE_ARG[@]+"${EXCLUDE_ARG[@]}"}
          "$ARRAY_SBATCH")
    if [[ "$DRY_RUN" == true ]]; then
        # %q-quote: the default constraint contains a `|`, and an unquoted echo of it would be
        # a pipe if anyone pasted this preview straight into a shell.
        printf '%q ' "${cmd[@]}" >&2; printf '\n' >&2
        echo "DRYRUN_${stage}"
    else
        "${cmd[@]}"
    fi
}

# pretrain goes first and alone: warm cells load its checkpoints, so its job id is the
# dependency everything warm is gated on.
PRETRAIN_ID=""
JOB_IDS=()
while read -r stage spec mem walltime n dep; do
    [[ "$stage" == "pretrain" ]] || continue
    PRETRAIN_ID=$(submit "$stage" "$spec" "$mem" "$walltime" "" "")
    JOB_IDS+=("$PRETRAIN_ID")
    echo "Submitted pretrain $spec ($n tasks, ${mem}G, $walltime) -> job $PRETRAIN_ID"
done <<< "$SPECS"

if [[ -z "$PRETRAIN_ID" ]]; then
    echo "ERROR: no pretrain stage in the grid -- warm cells would have nothing to start from." >&2
    exit 1
fi

while read -r stage spec mem walltime n dep; do
    [[ "$stage" == "pretrain" ]] && continue
    gate=""
    [[ "$dep" == "pretrain" ]] && gate="$PRETRAIN_ID"
    id=$(submit "$stage" "$spec" "$mem" "$walltime" "--nice" "$gate")
    JOB_IDS+=("$id")
    if [[ -n "$gate" ]]; then
        echo "Submitted $stage $spec ($n tasks, ${mem}G, $walltime, afterok:$gate) -> job $id"
    else
        echo "Submitted $stage $spec ($n tasks, ${mem}G, $walltime) -> job $id"
    fi
done <<< "$SPECS"

dependency=$(IFS=:; echo "${JOB_IDS[*]}")
if [[ "$DRY_RUN" == true ]]; then
    printf '%q ' sbatch --parsable --export=ALL,L96_EXP="$EXP" \
        --dependency=afterok:"$dependency" "$FINALIZE_SBATCH" >&2; printf '\n' >&2
    echo
    echo "(--dry-run: nothing was submitted)"
    exit 0
fi

finalize_id=$(sbatch --parsable --export=ALL,L96_EXP="$EXP" \
              --dependency=afterok:"$dependency" \
              ${EXCLUDE_ARG[@]+"${EXCLUDE_ARG[@]}"} "$FINALIZE_SBATCH")
echo "Submitted finalize (afterok:$dependency) -> job $finalize_id"

cat <<EOF

Monitor with:
  squeue -u \$USER
  sacct -j ${JOB_IDS[0]} --format=JobID,JobName%30,MaxRSS,Elapsed,State

On success, job $finalize_id writes $EXP/{runs,trace,diag}.csv and prints the R3
well-posedness check.

If ANY task fails, afterok is never satisfied and the dependent jobs sit in the queue forever
with reason DependencyNeverSatisfied. In that case:
  scancel $finalize_id
  python aggregate_shift.py --exp-dir $EXP     # names the missing task ids
  sbatch --export=ALL,L96_EXP=$EXP --array=<failed ids> --mem=<G> --time=<hms> $ARRAY_SBATCH
  sbatch --export=ALL,L96_EXP=$EXP $FINALIZE_SBATCH
EOF
