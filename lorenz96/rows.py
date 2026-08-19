import csv
from pathlib import Path

from .config import flat_columns

SCHEMA_VERSION = 1

_META = ["run_id", "run_label", "schema_version", "config_hash", "git_sha", "git_dirty", "created_utc", "host",
         "python_version", "torch_version", "numpy_version", "device", "determinism_enforced"]

_SEEDS = ["seed", "data_seed", "noise_seed", "init_seed", "shuffle_seed"]

_SHAPE = ["n_train", "n_val", "n_params", "chunks_used"]

_WARM = ["init_from", "init_ckpt_sha", "init_ckpt_step", "carry_optimizer",
         "intervention_kind", "intervention_params",
         "mix_ratio_requested", "mix_ratio_realized", "mix_new_per_batch", "mix_old_per_batch"]

_RESULT = [
    "status", "error", "wall_seconds", "steps_run", "samples_seen", "epochs_equiv",
    "train_loss_final", "train_loss_ema_final",
    # Both selection criteria are logged unconditionally: C1 (clean vs noisy target) then stays
    # open without re-running anything.
    "best_val_clean", "step_of_best_clean", "lr_at_best_clean", "best_at_boundary_clean",
    "best_val_noisy", "step_of_best_noisy", "lr_at_best_noisy", "best_at_boundary_noisy",
    # `next` is the selection criterion: predicted next state vs true next state. The other
    # two are kept so a run can be re-read under either, and so the clean-tendency artifact
    # stays visible rather than being quietly dropped from the record.
    "best_val_next", "step_of_best_next", "lr_at_best_next", "best_at_boundary_next",
    "final_val_clean", "final_val_noisy", "final_val_next",
    # Old-regime validation under data.source='shift'. *_at_best is the old-regime loss at the
    # step chosen by best_val_clean, i.e. what the selected checkpoint forgot.
    "final_val_old_clean", "final_val_old_noisy", "final_val_next_old",
    "val_old_clean_at_best", "val_old_noisy_at_best", "val_next_old_at_best",
    # Architecture-independent diagnostics only; per-layer norms go to diag.csv, whose long
    # format keeps this header from depending on n_layers.
    "diag_effective_rank_final", "diag_dormant_frac_final", "diag_weight_norm_final",
    "saved_tags", "ckpt_dir",
]

RUN_COLUMNS = _META + _SEEDS + flat_columns() + _SHAPE + _WARM + _RESULT

# One row per (run, eval point). lr_realized is logged because T8 records that 93% of round 1's
# ramp cycles peaked inside the warmup, so the labelled LR was not the LR that ran.
TRACE_COLUMNS = ["run_id", "step", "lr_realized", "train_loss_ema",
                 "val_clean", "val_noisy", "val_next",
                 "val_old_clean", "val_old_noisy", "val_next_old"]

# Long format: metrics whose names depend on the architecture (per-layer norms) or on the
# intervention (snp_norm_ratio) cannot live in a fixed-width header.
DIAG_COLUMNS = ["run_id", "step", "phase", "metric", "value"]


def append(path, columns, rows):
    # A declared header, checked on every append. Round 1 produced two result sets with
    # different schemas and no way to reconcile them (P2); this makes that a hard failure.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(columns)
    if path.exists() and path.stat().st_size:
        with open(path, newline="") as fh:
            existing = next(csv.reader(fh), [])
        if existing != header:
            missing = [c for c in header if c not in existing]
            extra = [c for c in existing if c not in header]
            raise ValueError(
                f"{path}: header mismatch. Appending would produce one file with two schemas.\n"
                f"  columns the code adds: {missing}\n"
                f"  columns only on disk:  {extra}\n"
                "Write to a new file, or migrate the old one deliberately.")
    else:
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(header)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="raise", restval="")
        for row in rows:
            unknown = [k for k in row if k not in header]
            if unknown:
                raise ValueError(f"{path}: row has columns absent from the declared header: "
                                 f"{unknown}")
            w.writerow(row)
    return path
