#!/usr/bin/env python
"""cold_new vs warm_new: forecast error vs lead time, with the paired per-seed ratio.

    python plot_pair.py --exp-dir results/shift

Both arms train on new-regime data only; the single difference is where the weights started.
The means alone would mislead here -- their seed ranges overlap at every lead -- so the lower
strip carries the PAIRED ratio. cold_new_s0 and warm_new_s0 share a data seed, an eval seed
and an init seed, so the per-seed ratio cancels the trajectory-to-trajectory variance that
swamps the unpaired comparison.
"""
import argparse
import collections
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Slots 2 and 3 of the dataviz reference palette, matching the stage colours used in
# figures/shift_new_regime_mse.png: cold = orange, warm = aqua.
COLOR = {"cold_new": "#eb6834", "warm_new": "#1baf7a"}
LABEL = {"cold_new": "cold_new  (from scratch)", "warm_new": "warm_new  (from pretrain)"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"
SAT_MSE = 2.0 * (4.38 / 3.64) ** 2      # two unrelated states from the F=10 attractor


def load(path, arms):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["arm"] in arms:
                d[r["arm"]][int(r["seed"])][int(r["lead_steps"])] = (
                    float(r["mse"]), float(r["days"]), float(r["nrmse"]))
    out = {}
    for arm, byseed in d.items():
        seeds = sorted(byseed)
        leads = sorted(byseed[seeds[0]])
        days = np.array([byseed[seeds[0]][k][1] for k in leads])
        mse = np.array([[byseed[s][k][0] for k in leads] for s in seeds])
        nrmse = np.array([[byseed[s][k][2] for k in leads] for s in seeds])
        out[arm] = (days, mse, nrmse)   # mse/nrmse are (n_seeds, n_leads)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp-dir", default="results/shift")
    ap.add_argument("--out", default="figures/shift_cold_vs_warm_new.png")
    args = ap.parse_args()

    d = load(Path(args.exp_dir) / "rollout_new.csv", set(COLOR))
    days = d["cold_new"][0]

    fig, (ax, rx) = plt.subplots(2, 1, figsize=(8.4, 6.6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.6, 1], hspace=0.12))
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, rx):
        a.set_facecolor(SURFACE)

    for arm in ("cold_new", "warm_new"):
        _, mse, _ = d[arm]
        for row in mse:                                  # every seed, not just the mean
            ax.plot(days, row, color=COLOR[arm], lw=0.9, alpha=0.30, zorder=2)
        ax.plot(days, mse.mean(axis=0), color=COLOR[arm], lw=2.2, zorder=3,
                label=LABEL[arm], solid_capstyle="round")

    ax.axhline(SAT_MSE, color=GRID, lw=1.2, zorder=1)
    ax.text(0.99, SAT_MSE * 1.08, "climatological saturation", ha="right", va="bottom",
            fontsize=8, color=MUTED, transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_ylabel("forecast MSE on new regime", fontsize=9.5, color=MUTED)
    # Two nearly-coincident curves: a legend reads better here than direct labels, which would
    # collide at every lead where the curves are close (which is all of them).
    ax.legend(frameon=False, fontsize=9.5, labelcolor=MUTED, loc="upper left",
              handlelength=1.8, borderaxespad=0.9)

    ratio = d["warm_new"][1] / d["cold_new"][1]          # paired: seed i over seed i
    for row in ratio:
        rx.plot(days, row, color=COLOR["warm_new"], lw=0.9, alpha=0.30, zorder=2)
    rx.plot(days, ratio.mean(axis=0), color=COLOR["warm_new"], lw=2.2, zorder=3)
    rx.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)

    horizon = days[np.argmax(d["cold_new"][2].mean(axis=0) > 0.3)]
    for a, tag in ((ax, False), (rx, True)):
        a.axvspan(horizon, days[-1] * 1.15, color="#000000", alpha=0.045, lw=0, zorder=0)
    ax.text(horizon * 1.06, 1.7e-3, "beyond useful horizon\n(NRMSE > 0.3): both arms are\n"
            "saturating into climatology,\nso the ratio is noise",
            fontsize=8, color=MUTED, ha="left", va="bottom")
    rx.text(0.985, 0.985, "warm_new worse", transform=rx.transAxes, ha="right", va="top",
            fontsize=8, color=MUTED)
    rx.text(0.985, 0.03, "warm_new better", transform=rx.transAxes, ha="right", va="bottom",
            fontsize=8, color=MUTED)
    rx.set_ylabel("paired ratio\nwarm ÷ cold", fontsize=9, color=MUTED)
    rx.set_ylim(0.78, 1.14)
    rx.set_xlabel("forecast lead time (days)", fontsize=9.5, color=MUTED)

    for a in (ax, rx):
        a.set_xscale("log")
        a.set_xticks([0.25, 0.5, 1, 2, 5, 10, 15])
        a.set_xticklabels(["6h", "12h", "1", "2", "5", "10", "15"])
        a.set_xticks([], minor=True)
        a.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=9, length=0)

    ax.set_title("Same data, different starting weights", fontsize=13, color=INK,
                 loc="left", pad=26)
    ax.text(0.0, 1.035, "both arms train on new-regime data only (mix ratio 1.0); "
            "thin lines are the 5 seeds", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color=MUTED)

    fig.subplots_adjust(left=0.115, right=0.975, top=0.86, bottom=0.095)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
