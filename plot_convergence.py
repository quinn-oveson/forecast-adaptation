#!/usr/bin/env python
"""Training curves for cold_new vs warm_new: is the arm gap a convergence artefact?

    python plot_convergence.py --exp-dir results/shift

Top panel is validation MSE on the NEW regime at every eval point; bottom is the training
loss over the same steps. Together they separate "not trained long enough" from "trained far
past the optimum" -- the two readings that a single end-of-training number cannot distinguish.

trace.val_next is the SELECTION set, which for every arm except pretrain is the new regime
(run_shift.py:148). Both arms here are non-pretrain, so no swap is needed -- but check that
before adding any arm to this figure.
"""
import argparse
import collections
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting import figure_path

COLOR = {"cold_new": "#eb6834", "warm_new": "#1baf7a"}
LABEL = {"cold_new": "cold_new  (from scratch)", "warm_new": "warm_new  (from pretrain)"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"


def load(exp):
    tr = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(exp / "trace.csv") as fh:
        for r in csv.DictReader(fh):
            arm = r["run_id"].rsplit("_s", 1)[0]
            if arm in COLOR:
                tr[arm][r["run_id"]].append(
                    (int(r["step"]), float(r["val_next"]), float(r["train_loss_ema"])))
    best = {}
    with open(exp / "runs.csv") as fh:
        for r in csv.DictReader(fh):
            if r["run_label"] in COLOR:
                best[r["run_id"]] = (int(r["step_of_best_next"]),
                                     float(r["best_val_next"]),
                                     r["best_at_boundary_next"] == "True")
    return tr, best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp-dir", default="results/shift")
    ap.add_argument("--out", default=None,
                    help="override the default figures/<exp-dir-name>/<name>.png")
    args = ap.parse_args()
    out = figure_path(args.exp_dir, "convergence_new", args.out)
    exp = Path(args.exp_dir)
    tr, best = load(exp)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.55, 1], hspace=0.13))
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, bx):
        a.set_facecolor(SURFACE)

    for arm, runs in tr.items():
        for i, (run, pts) in enumerate(sorted(runs.items())):
            pts.sort()
            step = np.array([p[0] for p in pts])
            ax.plot(step, [p[1] for p in pts], color=COLOR[arm], lw=1.3, alpha=0.55,
                    zorder=3, label=LABEL[arm] if i == 0 else None)
            bx.plot(step, [p[2] for p in pts], color=COLOR[arm], lw=1.3, alpha=0.55, zorder=3)
            s_best, v_best, _ = best[run]
            ax.scatter([s_best], [v_best], s=58, color=COLOR[arm], edgecolors=SURFACE,
                       linewidths=1.8, zorder=5)

    for arm, xtext, ha in (("warm_new", 22, "left"), ("cold_new", 6400, "right")):
        med = int(np.median([best[r][0] for r in tr[arm]]))
        ax.annotate(f"best ≈ step {med}", xy=(med, min(best[r][1] for r in tr[arm])),
                    xytext=(xtext, 7.9e-4), ha=ha, fontsize=8.5, color=COLOR[arm],
                    arrowprops=dict(arrowstyle="-", color=COLOR[arm], lw=1.0, alpha=0.6))

    # 4 of 5 warm_new seeds take a late loss spike that partly undoes the memorisation:
    # training loss jumps ~50x and validation RECOVERS. Real, and visible in both panels.
    ax.annotate("late instability:\n4/5 warm seeds partly\nrecover from overfitting",
                xy=(5000, 5.5e-3), xytext=(90, 1.75e-2), fontsize=8.5, color=MUTED,
                ha="center", arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0,
                                             alpha=0.7, connectionstyle="arc3,rad=-0.25"))

    ax.set_yscale("log")
    ax.set_ylabel("validation MSE, new regime\n(one-step / next-state)", fontsize=9.5,
                  color=MUTED)
    ax.set_ylim(7.2e-4, 2.4e-2)

    ax.set_title("Both arms overfit; neither is short of convergence", fontsize=13,
                 color=INK, loc="left", pad=26)
    ax.text(0.0, 1.035, "each line is one seed; dots mark the selected checkpoint "
            "(best_next)", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=MUTED)

    bx.set_yscale("log")
    bx.set_ylabel("training loss (EMA)", fontsize=9.5, color=MUTED)
    bx.set_xlabel("training step  (budget = 6000)", fontsize=9.5, color=MUTED)
    h, l = ax.get_legend_handles_labels()
    bx.legend(h, l, frameon=False, fontsize=9.5, labelcolor=MUTED, loc="lower left",
              handlelength=1.8, borderaxespad=0.9)

    for a in (ax, bx):
        a.set_xscale("log")
        a.set_xlim(1, 6600)
        a.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9, length=0)

    fig.subplots_adjust(left=0.135, right=0.975, top=0.865, bottom=0.085)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
