#!/bin/bash --login
# Post-processing for the forcing-shift experiment: join the per-task CSVs and report.
#
# Not meant to be submitted by hand -- slurm/submit_shift_experiment.sh submits this last with
# --dependency=afterok on every training array, so it runs only once all 45 cells have
# succeeded. Running it by hand afterwards is fine too; it is cheap and idempotent.
#
# CPU-only and short: this reads CSVs and writes CSVs. No training, no GPU.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH -J l96_shift_finalize
#SBATCH --output=slurm_logs/finalize_shift_%j.out
#SBATCH --error=slurm_logs/finalize_shift_%j.err

set -euo pipefail

: "${L96_EXP:?not set -- submit with --export=ALL,L96_EXP=results/shift}"

mamba activate forecast-adaptation   # same env as shift_array.sbatch

cd "$SLURM_SUBMIT_DIR"   # repo root -- NOT $(dirname "$0"), SLURM copies this script to
                          # /var/spool/slurmd before running it
export PYTHONUNBUFFERED=1

python aggregate_shift.py --exp-dir "$L96_EXP"

echo
echo "Copy these back to your laptop to look at them:"
echo "  $L96_EXP/runs.csv    one row per cell"
echo "  $L96_EXP/trace.csv   validation curve in both regimes, per eval point"
echo "  $L96_EXP/diag.csv    per-layer norms and plasticity diagnostics, long format"
echo
echo "Checkpoints stay on the cluster ($L96_EXP/checkpoints/, ~32 MB each at hidden=512)."
echo "Forecast metrics are NOT computed here -- train.py logs losses only. The evaluator that"
echo "turns these checkpoints into NRMSE and valid prediction time does not exist yet, and"
echo "metrics.nrmse_curve still has the M4 denominator bug, which matters precisely because"
echo "this experiment compares across two forcings."
