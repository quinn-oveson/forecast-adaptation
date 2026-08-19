#!/usr/bin/env python
"""Autoregressive forecast error vs lead time, per arm, on the post-shift regime.

    python rollout_curves.py --exp-dir results/shift        # computes, caches to rollout.csv

This is the evaluator finalize_shift.sbatch says does not exist: train.py logs one-step losses
only, and trace.csv is validation-vs-training-step, not validation-vs-lead-time. Here each
saved checkpoint is rolled out on its own held-out trajectory and scored at every lead.

One model step is STRIDE * DT = 0.05 MTU. With 1 MTU ~= 5 days that is 6 h, so `lead_steps` k
is a k*6 h forecast.

Every run is evaluated at run_seed + EVAL_SEED_OFFSET, the band reserved in metrics.py, so no
rollout can land on a trajectory the model trained on.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from lorenz96 import checkpoint as ckpt_mod
from lorenz96.metrics import EVAL_SEED_OFFSET, evaluate
from lorenz96.models import CircularCNN
from run_shift import GRID_SNAPSHOT, build_cells, run_id_of

FIELDS = ["arm", "seed", "lead_steps", "mtu", "days", "mse", "nrmse"]


def curve_for(ckpt_path, F, seed, n_steps, n_init, init_noise, device="cpu"):
    ck = ckpt_mod.load(ckpt_path)
    arch = ck["arch"]
    model = CircularCNN(hidden=arch["hidden"], n_layers=arch["n_layers"],
                        kernel=arch["kernel"], in_channels=arch["in_channels"],
                        heteroscedastic=arch["heteroscedastic"])
    model.load_state_dict(ck["model"])
    # evaluate() reads the device off the model's parameters, so this is what selects it.
    # Backends do not agree bit-for-bit in float32, and a 60-step autoregressive rollout on a
    # chaotic system compounds that -- keep one device for a whole figure.
    model.to(device)
    res = evaluate(model, F=F, n_steps=n_steps, n_init=n_init, seed=seed,
                   history=arch["in_channels"], init_noise=init_noise)
    # nrmse_curve divides RMSE by sqrt(2) * clim, and clim is exactly 1.0 now that the
    # denominator is frozen at the reference climatology -- so MSE comes back exactly.
    nrmse = res["curve"].cpu().numpy()
    return nrmse, 2.0 * nrmse ** 2


def compute(exp, out_csv, n_steps, n_init, device):
    cfg = yaml.safe_load(open(exp / "config.resolved.yaml"))
    grid = yaml.safe_load(open(exp / GRID_SNAPSHOT))
    F_new, noise = cfg["data"]["F_new"], cfg["data"]["noise"]
    from lorenz96.data import DT, STRIDE
    step_mtu = STRIDE * DT

    cells = build_cells(**grid)
    fh = open(out_csv, "w", newline="")
    w = csv.DictWriter(fh, FIELDS)
    w.writeheader()
    n_rows = 0
    for i, cell in enumerate(cells, 1):
        path = exp / "checkpoints" / run_id_of(cell) / "best_next.pt"
        if not path.exists():
            print(f"  skip {run_id_of(cell)} (no checkpoint)")
            continue
        nrmse, mse = curve_for(path, F_new, cell.seed + EVAL_SEED_OFFSET, n_steps, n_init,
                               noise, device)
        for k in range(len(mse)):
            lead = k + 1
            w.writerow(dict(arm=cell.arm, seed=cell.seed, lead_steps=lead,
                            mtu=lead * step_mtu, days=lead * step_mtu * 5.0,
                            mse=float(mse[k]), nrmse=float(nrmse[k])))
            n_rows += 1
        fh.flush()
        print(f"  [{i:2d}/{len(cells)}] {run_id_of(cell):26s} "
              f"mse@6h={mse[0]:.3e}  mse@5d={mse[min(19, len(mse) - 1)]:.3e}", flush=True)

    fh.close()
    print(f"\nwrote {out_csv}  ({n_rows} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp-dir", default="results/shift")
    ap.add_argument("--n-steps", type=int, default=60, help="leads to roll out (60 = 15 days)")
    ap.add_argument("--n-init", type=int, default=64)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, reproducible), or mps/cuda for speed")
    args = ap.parse_args()
    exp = Path(args.exp_dir)
    torch.set_grad_enabled(False)
    print(f"device: {args.device}")
    compute(exp, exp / "rollout_new.csv", args.n_steps, args.n_init, args.device)


if __name__ == "__main__":
    main()
