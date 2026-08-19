#!/usr/bin/env python
"""Drive the uniform forcing-shift adaptation experiment.

A large pre-shift pool at F_old and a small post-shift pool at F_new. Every arm is a
composition of (init source, mix_ratio), so they differ only in those two axes:

    pretrain        scratch, ratio 0.0        the pre-shift model; also the `frozen` baseline
    cold_all        scratch, ratio natural    retrain from scratch on everything
    cold_new        scratch, ratio 1.0        retrain from scratch on the scarce new data only
    warm_replay_R   pretrain, ratio R         fine-tune with replay at R (R=1.0 is no replay)
    cold_all_conv   scratch, ratio natural    cold_all with a larger budget, run only as a
                                              ceiling and reported separately (T7)

Every matched arm gets the same step budget, batch size and samples-seen; per-batch mixing
(shift.MixSampler) means the ratio is the only quantity that varies across the replay sweep.

The pre-shift model is selected on OLD-regime validation. Selecting it on the new regime would
leak the shift into a model that is supposed to predate it. Its new-regime loss is still logged
every step, which is what makes `frozen` available for the R3 well-posedness check at no cost.

Two ways to run it:

  laptop      python run_shift.py --config cfg/shift.yaml --seeds 0 1 2
              one process, all cells, stream built once per seed and reused across arms

  cluster     python run_shift.py --config cfg/shift.yaml --seeds 0 1 2 3 4 --freeze
              python run_shift.py --exp-dir results/shift --task-id N     (one cell per task)
              see slurm/ -- --freeze writes the resolved config and grid that every task reads

Cells are ordered stage-major (pretrain, cold, conv, warm) so each stage is a contiguous
--array range, and every stage's ids come from --print-array-specs rather than being written
out by hand.
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from lorenz96 import checkpoint as ckpt_mod
from lorenz96 import config as cfgmod
from lorenz96 import rows
from lorenz96.shift import MixSampler, resolve_ratio
from train import build_stream, train_once

DEFAULT_REPLAY = ["natural", 0.25, 0.5, 0.75, 1.0]
PRETRAIN = "pretrain"
STAGES = ("pretrain", "cold", "conv", "warm")
CONFIG_SNAPSHOT = "config.resolved.yaml"
GRID_SNAPSHOT = "grid.yaml"

# Measured from dry run 
TIER_RESOURCES = {"pretrain": (4, "00:10:00"), "cold": (4, "00:10:00"),
                  "warm": (4, "00:10:00"), "conv": (4, "00:30:00")}


@dataclass(frozen=True)
class Cell:
    task_id: int
    stage: str
    arm: str
    seed: int
    ratio: Any
    init_arm: Optional[str]
    budget_mult: int


def replay_label(ratio):
    return "warm_new" if ratio == 1.0 else f"warm_replay_{ratio}"


def stage_arms(replay, conv_mult, pretrain_mult):
    # (arm, ratio, init_arm, budget_mult) per stage. Stage order fixes the task_id layout.
    return {
        "pretrain": [(PRETRAIN, 0.0, None, pretrain_mult)],
        "cold": [("cold_all", "natural", None, 1), ("cold_new", 1.0, None, 1)],
        "conv": ([("cold_all_conv", "natural", None, conv_mult)] if conv_mult > 1 else []),
        "warm": [(replay_label(r), r, PRETRAIN, 1) for r in replay],
    }


def build_cells(replay, seeds, conv_mult, pretrain_mult):
    arms, cells, task_id = stage_arms(replay, conv_mult, pretrain_mult), [], 0
    for stage in STAGES:
        for arm, ratio, init_arm, mult in arms[stage]:
            for seed in seeds:
                cells.append(Cell(task_id, stage, arm, seed, ratio, init_arm, mult))
                task_id += 1
    return cells


def decode_task_id(task_id, grid):
    cells = build_cells(**grid)
    if not 0 <= task_id < len(cells):
        raise SystemExit(f"task_id {task_id} outside 0..{len(cells) - 1} for this grid")
    return cells[task_id]


def array_specs(grid):
    # One line per stage: stage, contiguous array range, mem, walltime, task count, dependency.
    cells = build_cells(**grid)
    out = []
    for stage in STAGES:
        ids = [c.task_id for c in cells if c.stage == stage]
        if not ids:
            continue
        mem, walltime = TIER_RESOURCES[stage]
        depends = "pretrain" if stage == "warm" else "none"
        out.append((stage, f"{min(ids)}-{max(ids)}", mem, walltime, len(ids), depends))
    return out


def cell_config(config_path, cell, out_dir, overrides=()):
    ov = list(overrides) + [("seed", cell.seed), ("data.mix_ratio", cell.ratio),
                            ("io.out_dir", out_dir)]
    if cell.init_arm:
        ov.append(("io.init_from", str(ckpt_path(out_dir, cell.init_arm, cell.seed))))
    cfg = cfgmod.load(config_path, ov)
    if cell.budget_mult != 1:
        ov.append(("train.budget_steps", cfg.train.budget_steps * cell.budget_mult))
        cfg = cfgmod.load(config_path, ov)
    return cfg


def run_id_of(cell):
    return f"{cell.arm}_s{cell.seed}"


def ckpt_path(out_dir, arm, seed):
    return Path(out_dir) / "checkpoints" / f"{arm}_s{seed}" / "best_next.pt"


def run_cell(cfg, cell, stream, out_dir, writer):
    run_id = run_id_of(cell)
    seeds = cfgmod.resolved_seeds(cfg)
    init_from = cfg.io.init_from
    if init_from and not Path(init_from).exists():
        raise SystemExit(
            f"{run_id}: needs {init_from}, which does not exist. Warm arms depend on the "
            "pretrain stage; submit it first and gate this array on --dependency=afterok.")

    ratio = resolve_ratio(cfg.data.mix_ratio, stream.n_old, stream.n_new)
    sampler = MixSampler(stream.n_old, stream.n_new, ratio, cfg.train.batch_size,
                         torch.Generator().manual_seed(seeds["shuffle_seed"]))
    # The pre-shift model never selects on post-shift data; see the module docstring.
    select, other = ((stream.val_old, stream.val_new) if cell.arm == PRETRAIN
                     else (stream.val_new, stream.val_old))

    result = train_once(cfg, stream.pooled(), select, run_id=run_id,
                        init_ckpt=ckpt_mod.load(init_from) if init_from else None,
                        init_from=init_from, val_old=other, sampler=sampler)
    result.run["run_label"] = cell.arm
    result.run["ckpt_dir"] = str(Path(out_dir) / "checkpoints" / run_id)

    for tag, snap in result.snapshots.items():
        ckpt_mod.save(Path(result.run["ckpt_dir"]) / f"{tag}.pt", snap, cfg,
                      cfgmod.provenance(), tag, seeds)
    writer(result)

    # For every arm but pretrain, `new` is the selection regime and `old` is the forgetting
    # measure; for pretrain the two are swapped, so report them by regime, not by role.
    swap = cell.arm == PRETRAIN
    new_loss = result.run["val_next_old_at_best"] if swap else result.run["best_val_next"]
    old_loss = result.run["best_val_next"] if swap else result.run["val_next_old_at_best"]
    print(f"  {run_id:24s} rho={ratio:<7.4f} new={new_loss:.6f} old={old_loss:.6f} "
          f"@{result.run['step_of_best_next']:<5d} {result.run['wall_seconds']:.0f}s",
          flush=True)
    return new_loss, old_loss


def shared_writer(out_dir):
    def write(result):
        rows.append(Path(out_dir) / "runs.csv", rows.RUN_COLUMNS, [result.run])
        rows.append(Path(out_dir) / "trace.csv", rows.TRACE_COLUMNS, result.trace)
        if result.diag:
            rows.append(Path(out_dir) / "diag.csv", rows.DIAG_COLUMNS, result.diag)
    return write


def task_writer(out_dir, task_id):
    # One set of files per task. 45 array tasks appending to one runs.csv would interleave
    # partial lines; per-task files also make a single failed task re-runnable on its own.
    # Truncated on open, so a re-run overwrites cleanly and requeue-on-preemption is safe.
    def write(result):
        base = Path(out_dir) / "tasks"
        for name, cols, data in (("runs", rows.RUN_COLUMNS, [result.run]),
                                 ("trace", rows.TRACE_COLUMNS, result.trace),
                                 ("diag", rows.DIAG_COLUMNS, result.diag)):
            path = base / f"task{task_id:03d}_{name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
            rows.append(path, cols, data)
    return write


def report(seed, summary):
    frozen, cold_all = summary.get(PRETRAIN), summary.get("cold_all")
    print(f"\n  seed {seed}: R3 well-posedness check")
    if not (frozen and cold_all):
        print("    skipped (need both pretrain and cold_all in this run)")
        return
    ratio = frozen[0] / cold_all[0] if cold_all[0] else float("nan")
    verdict = ("separated" if ratio > 1.05 else
               "NOT SEPARATED -- no arm contrast here is interpretable (R3)")
    print(f"    frozen (pre-shift model on new regime): {frozen[0]:.6f}")
    print(f"    cold_all on new regime:                 {cold_all[0]:.6f}")
    print(f"    frozen / cold_all = {ratio:.3f}   {verdict}")


def freeze(args, cfg, grid, out_dir):
    out = Path(out_dir)
    stale = sorted(out.glob("tasks/task*_runs.csv"))
    if stale and not args.overwrite:
        raise SystemExit(
            f"{out}/tasks already holds {len(stale)} task result files from an earlier run. "
            "Aggregation would mix them into the new summary. Move them aside, or pass "
            "--overwrite if they are meant to be replaced.")
    cfgmod.dump(cfg, out / CONFIG_SNAPSHOT)
    with open(out / GRID_SNAPSHOT, "w") as fh:
        yaml.safe_dump(grid, fh, sort_keys=False)
    print(f"froze {out / CONFIG_SNAPSHOT}")
    print(f"froze {out / GRID_SNAPSHOT}")
    print_specs(grid)


def print_specs(grid):
    for row in array_specs(grid):
        print(" ".join(str(v) for v in row))


def load_frozen(exp_dir):
    exp = Path(exp_dir)
    for name in (CONFIG_SNAPSHOT, GRID_SNAPSHOT):
        if not (exp / name).exists():
            raise SystemExit(f"{exp / name} missing -- run --freeze before submitting tasks.")
    grid = yaml.safe_load(open(exp / GRID_SNAPSHOT))
    return exp / CONFIG_SNAPSHOT, grid


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="cfg/shift.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--replay", nargs="+", default=DEFAULT_REPLAY)
    ap.add_argument("--conv-mult", type=int, default=10,
                    help="budget multiplier for the cold_all_conv ceiling; 1 disables the stage")
    ap.add_argument("--pretrain-mult", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="DOTTED=VALUE")
    ap.add_argument("--freeze", action="store_true",
                    help="write the resolved config and grid a cluster array will read")
    ap.add_argument("--exp-dir", default=None,
                    help="read the frozen config and grid from here (--task-id mode)")
    ap.add_argument("--task-id", type=int, default=None, help="run exactly one cell")
    ap.add_argument("--print-array-specs", action="store_true")
    ap.add_argument("--list-cells", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    overrides = [tuple(s.split("=", 1)) for s in args.set]

    # --exp-dir means the grid and config are already frozen; anything on the command line that
    # would change them is ignored on purpose, so a queued task cannot drift from what was
    # submitted.
    if args.exp_dir:
        config_path, grid = load_frozen(args.exp_dir)
        out_dir = args.exp_dir
    else:
        config_path = args.config
        grid = dict(replay=[r if r == "natural" else float(r) for r in args.replay],
                    seeds=list(args.seeds), conv_mult=args.conv_mult,
                    pretrain_mult=args.pretrain_mult)
        out_dir = args.out_dir or cfgmod.load(config_path, overrides).io.out_dir

    cells = build_cells(**grid)

    if args.print_array_specs:
        print_specs(grid)
        return 0
    if args.list_cells:
        for c in cells:
            print(f"{c.task_id:3d} {c.stage:9s} {c.arm:22s} seed={c.seed} ratio={c.ratio} "
                  f"init={c.init_arm or '-'} budget_x{c.budget_mult}")
        return 0
    if args.freeze:
        freeze(args, cfgmod.load(config_path, overrides), grid, out_dir)
        return 0

    if args.dry_run:
        base = cfgmod.load(config_path, overrides)
        print(f"config      {config_path}")
        print(f"regimes     F_old={base.data.F_old} -> F_new={base.data.F_new}   "
              f"n_old={base.data.n_old} n_new={base.data.n_new}")
        print(f"grid        {grid}")
        print(f"cells       {len(cells)}")
        for stage, spec, mem, walltime, n, dep in array_specs(grid):
            print(f"  {stage:9s} array={spec:8s} mem={mem}G time={walltime} "
                  f"tasks={n} depends={dep}")
        print(f"total steps {sum(base.train.budget_steps * c.budget_mult for c in cells):,}")
        return 0

    if args.task_id is not None:
        cell = decode_task_id(args.task_id, grid)
        cfg = cell_config(config_path, cell, out_dir, overrides)
        stream = build_stream(cfg, cfgmod.resolved_seeds(cfg))
        print(f"task {args.task_id}: {cell.stage}/{cell.arm} seed={cell.seed}", flush=True)
        run_cell(cfg, cell, stream, out_dir, task_writer(out_dir, args.task_id))
        return 0

    write = shared_writer(out_dir)
    for seed in grid["seeds"]:
        print(f"\nseed {seed}", flush=True)
        summary, stream, stream_seed = {}, None, None
        for cell in [c for c in cells if c.seed == seed]:
            cfg = cell_config(config_path, cell, out_dir, overrides)
            if stream is None or stream_seed != seed:
                stream, stream_seed = build_stream(cfg, cfgmod.resolved_seeds(cfg)), seed
            summary[cell.arm] = run_cell(cfg, cell, stream, out_dir, write)
        report(seed, summary)
    print(f"\nwrote {out_dir}/runs.csv, trace.csv, diag.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
