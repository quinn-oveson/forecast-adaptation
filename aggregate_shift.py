#!/usr/bin/env python
"""Concatenate per-task CSVs from a cluster run into one set of tables, then report.

Array tasks each write results/<exp>/tasks/task<NNN>_{runs,trace,diag}.csv rather than
appending to a shared file, because 45 tasks appending to one runs.csv would interleave partial
lines. This joins them back together, refuses to do so if a header disagrees, and says which
task ids are missing instead of quietly summarising a partial grid.

Uses only csv and numpy: environment-cluster.yml has no pandas.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from lorenz96 import rows
from run_shift import GRID_SNAPSHOT, PRETRAIN, build_cells

KINDS = (("runs", rows.RUN_COLUMNS), ("trace", rows.TRACE_COLUMNS), ("diag", rows.DIAG_COLUMNS))


def collect(exp_dir, kind, columns):
    paths = sorted(Path(exp_dir).glob(f"tasks/task*_{kind}.csv"))
    merged, seen_header = [], None
    for path in paths:
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            if header != list(columns):
                missing = [c for c in columns if c not in header]
                extra = [c for c in header if c not in columns]
                raise SystemExit(
                    f"{path}: header does not match the declared {kind} schema.\n"
                    f"  columns the code expects but the file lacks: {missing}\n"
                    f"  columns in the file but not the schema:      {extra}\n"
                    "That file came from a different version of the code. Re-run that task or "
                    "archive the old results; do not merge two schemas into one table.")
            seen_header = header
            merged += [dict(zip(header, row)) for row in reader]
    return seen_header or list(columns), merged, paths


def completeness(exp_dir, grid):
    expected = {c.task_id: c for c in build_cells(**grid)}
    found = {int(p.name[4:7]) for p in Path(exp_dir).glob("tasks/task*_runs.csv")}
    missing = sorted(set(expected) - found)
    return expected, missing


def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def summarise(runs):
    # Report by regime, not by role: pretrain selects on the old regime, so its two validation
    # columns are swapped relative to every other arm.
    by_arm = defaultdict(list)
    for r in runs:
        swap = r["run_label"] == PRETRAIN
        new = fnum(r, "val_next_old_at_best" if swap else "best_val_next")
        old = fnum(r, "best_val_next" if swap else "val_next_old_at_best")
        by_arm[r["run_label"]].append((new, old, fnum(r, "step_of_best_next"),
                                       r.get("best_at_boundary_next", "")))
    return by_arm


def report(by_arm):
    print(f"\n{'arm':24s} {'new regime':>22s} {'old regime':>22s} {'step':>7s} {'n':>3s} bnd")
    print(f"{'':24s} {'mean +/- sd':>22s} {'mean +/- sd':>22s}")
    for arm in sorted(by_arm, key=lambda a: (a != PRETRAIN, a)):
        vals = by_arm[arm]
        new = np.array([v[0] for v in vals]); old = np.array([v[1] for v in vals])
        step = np.array([v[2] for v in vals])
        n_bnd = sum(str(v[3]).lower() == "true" for v in vals)
        print(f"{arm:24s} {np.nanmean(new):11.6f} +/-{np.nanstd(new):8.6f} "
              f"{np.nanmean(old):11.6f} +/-{np.nanstd(old):8.6f} "
              f"{np.nanmean(step):7.0f} {len(vals):3d} {n_bnd:3d}")
    print("\nbnd = cells whose optimum sat at the last eval point: the budget chose the model, "
          "not the data (C6).")


def well_posed(by_arm):
    print("\nR3 well-posedness check")
    if PRETRAIN not in by_arm or "cold_all" not in by_arm:
        print("  skipped (need both pretrain and cold_all)")
        return
    frozen = np.nanmean([v[0] for v in by_arm[PRETRAIN]])
    cold = np.nanmean([v[0] for v in by_arm["cold_all"]])
    ratio = frozen / cold if cold else float("nan")
    verdict = ("separated" if ratio > 1.05 else
               "NOT SEPARATED -- no arm contrast in this table is interpretable (R3)")
    print(f"  frozen (pre-shift model on new regime): {frozen:.6f}")
    print(f"  cold_all on new regime:                 {cold:.6f}")
    print(f"  frozen / cold_all = {ratio:.3f}   {verdict}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="summarise anyway when task result files are missing")
    args = ap.parse_args(argv)
    exp = Path(args.exp_dir)

    grid_path = exp / GRID_SNAPSHOT
    if grid_path.exists():
        expected, missing = completeness(exp, yaml.safe_load(open(grid_path)))
        print(f"{len(expected) - len(missing)}/{len(expected)} tasks present")
        if missing:
            print(f"MISSING task ids: {','.join(str(m) for m in missing)}")
            for t in missing[:10]:
                c = expected[t]
                print(f"  {t:3d} {c.stage:9s} {c.arm:22s} seed={c.seed}")
            if not args.allow_incomplete:
                raise SystemExit(
                    "Refusing to aggregate a partial grid. Re-run the missing tasks:\n"
                    f"  sbatch --array={','.join(str(m) for m in missing)} "
                    "slurm/shift_array.sbatch\n"
                    "or pass --allow-incomplete to summarise what is there.")
    else:
        print(f"no {GRID_SNAPSHOT} in {exp}; skipping the completeness check")

    runs = None
    for kind, columns in KINDS:
        header, merged, paths = collect(exp, kind, columns)
        if not paths:
            print(f"no task*_{kind}.csv found")
            continue
        out = exp / f"{kind}.csv"
        with open(out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, restval="")
            writer.writeheader()
            writer.writerows(merged)
        print(f"wrote {out}  ({len(merged)} rows from {len(paths)} tasks)")
        if kind == "runs":
            runs = merged

    if runs:
        errors = [r for r in runs if r.get("status") != "ok"]
        if errors:
            print(f"\n{len(errors)} cells recorded status != ok:")
            for r in errors:
                print(f"  {r['run_id']:24s} {r['error'][:90]}")
        by_arm = summarise([r for r in runs if r.get("status") == "ok"])
        report(by_arm)
        well_posed(by_arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
