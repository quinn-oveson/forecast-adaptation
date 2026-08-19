#!/usr/bin/env python
"""Forecast error vs lead time on the post-shift regime, per arm.

    python rollout_curves.py --exp-dir results/shift     # writes rollout_new.csv first
    python plot_rollout.py   --exp-dir results/shift

Two panels rather than nine lines on one axis: nine is past the categorical colour cap, and
the five replay arms are an ORDERED SWEEP, not nine unrelated identities. The sweep therefore
gets a single-hue ordinal ramp (magnitude), and the controls get categorical hues (identity).
"""
import argparse
import collections
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting import figure_path

# dataviz reference palette. Controls: categorical slots 1-4. Sweep: the documented blue
# ordinal ramp, steps 250-650 (nothing lighter than 250 on a light surface).
CONTROL_COLOR = {"pretrain": "#2a78d6", "cold_all": "#eb6834",
                 "cold_all_conv": "#1baf7a", "cold_new": "#eda100"}
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
WARM = [("warm_replay_natural", "0.09"), ("warm_replay_0.25", "0.25"),
        ("warm_replay_0.5", "0.50"), ("warm_replay_0.75", "0.75"), ("warm_new", "1.00")]
CONTROLS = [("pretrain", "pretrain (frozen)"), ("cold_all", "cold_all"),
            ("cold_all_conv", "cold_all_conv (10x)"), ("cold_new", "cold_new")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"

# Two unrelated states drawn from the F=10 attractor. sigma(F=10)/sigma(F=8) = 4.38/3.64 in
# frozen-normalised units (notes/TRANSFER.md), so saturation is 2 * that^2. Above this line a
# forecast carries no information the climatology does not already have.
SAT_MSE = 2.0 * (4.38 / 3.64) ** 2


def load(path):
    curves = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            curves[r["arm"]][float(r["days"])].append(float(r["mse"]))
    out = {}
    for arm, byday in curves.items():
        days = np.array(sorted(byday))
        out[arm] = (days, np.array([np.mean(byday[d]) for d in days]))
    return out


def panel(ax, curves, series, title, ref=()):
    for arm, label, color in series:
        if arm not in curves:
            continue
        d, m = curves[arm]
        ax.plot(d, m, color=color, lw=2.0, label=label, zorder=3, solid_capstyle="round")
    for arm, label in ref:
        if arm in curves:
            d, m = curves[arm]
            ax.plot(d, m, color=MUTED, lw=1.1, ls=(0, (4, 3)), alpha=0.6, zorder=2,
                    label=label)
    ax.axhline(SAT_MSE, color=GRID, lw=1.2, zorder=1)
    ax.text(0.985, SAT_MSE * 1.09, "climatological saturation", ha="right", va="bottom",
            fontsize=8, color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_xscale("log"); ax.set_yscale("log")
    # Plain day numbers: 10^0 / 10^1 is unreadable as a forecast horizon.
    ticks = [0.25, 0.5, 1, 2, 5, 10, 15]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["6h", "12h", "1", "2", "5", "10", "15"])
    ax.set_xticks([], minor=True)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlabel("forecast lead time (days)", fontsize=9.5, color=MUTED)
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="lower right",
              handlelength=1.6, borderaxespad=0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp-dir", default="results/shift")
    ap.add_argument("--out", default=None,
                    help="override the default figures/<exp-dir-name>/<name>.png")
    args = ap.parse_args()
    out = figure_path(args.exp_dir, "rollout_new_regime", args.out)

    curves = load(Path(args.exp_dir) / "rollout_new.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)

    panel(a1, curves, [(a, l, CONTROL_COLOR[a]) for a, l in CONTROLS],
          "Controls")
    panel(a2, curves, [(a, f"replay {l}", c) for (a, l), c in zip(WARM, RAMP)],
          "Warm start from pretrain, by replay ratio",
          ref=[("pretrain", "pretrain (do nothing)")])
    a1.set_ylabel("forecast MSE on new regime", fontsize=9.5, color=MUTED)

    fig.suptitle("Forecast error vs lead time, post-shift regime (F = 10)",
                 fontsize=13, color=INK, x=0.045, ha="left", y=0.975)
    fig.text(0.045, 0.905, "mean of 5 seeds; one model step = 6 h; "
             "1 Lyapunov time at F=10 ≈ 2.2 days", fontsize=9.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
