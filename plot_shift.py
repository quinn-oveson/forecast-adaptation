#!/usr/bin/env python
"""One-step (next-state) MSE on new-regime validation data, per arm.

    python plot_shift.py --exp-dir results/shift

Reads the frozen grid's runs.csv and plots every seed as its own point, because with n=5 a
mean alone hides whether an arm difference is bigger than the seed spread.

THE COLUMN NAMES ARE ROLE-BASED, NOT REGIME-BASED. `best_val_next` is the SELECTION set and
`val_next_old_at_best` is the other one -- and for the pretrain arm those are swapped relative
to every other arm, because a pre-shift model must not select its checkpoint on post-shift
data (run_shift.py:148). Plotting `best_val_next` straight would put pretrain at 7.7e-4 and
rank the do-nothing baseline as the best arm on the new regime. aggregate_shift.py:63 applies
the same swap; this file does not re-derive it, it imports it.
"""
import argparse
import collections
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_shift import PRETRAIN

# dataviz reference palette, categorical slots 1-3 (light mode). Three is the documented cap
# for all-pairs forms like this one; the triple is pre-validated, not eyeballed.
STAGE_COLOR = {"frozen": "#2a78d6", "cold": "#eb6834", "warm": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

# Display order: baseline first, then the two from-scratch controls, then the replay sweep by
# ascending new-regime fraction. Not alphabetical -- the sweep only reads as a sweep in order.
ARMS = [("pretrain", "pretrain (frozen)", "frozen"),
        ("cold_all", "cold_all", "cold"),
        ("cold_all_conv", "cold_all_conv (10x)", "cold"),
        ("cold_new", "cold_new", "cold"),
        ("warm_replay_natural", "warm replay natural (0.09)", "warm"),
        ("warm_replay_0.25", "warm replay 0.25", "warm"),
        ("warm_replay_0.5", "warm replay 0.5", "warm"),
        ("warm_replay_0.75", "warm replay 0.75", "warm"),
        ("warm_new", "warm_new (1.0)", "warm")]


def new_regime_mse(row):
    swap = row["run_label"] == PRETRAIN
    return float(row["val_next_old_at_best" if swap else "best_val_next"])


def load(exp_dir):
    by_arm = collections.defaultdict(list)
    with open(Path(exp_dir) / "runs.csv") as fh:
        for row in csv.DictReader(fh):
            if row["status"] == "ok":
                by_arm[row["run_label"]].append(new_regime_mse(row))
    return by_arm


def plot(by_arm, out_path, title_note=""):
    arms = [a for a in ARMS if a[0] in by_arm]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    rng = np.random.default_rng(0)   # jitter only, never touches a value
    for i, (arm, label, stage) in enumerate(arms):
        y = len(arms) - 1 - i
        v = np.array(by_arm[arm]) * 1e3
        color = STAGE_COLOR[stage]
        ax.scatter(v, y + rng.uniform(-0.15, 0.15, len(v)), s=34, color=color,
                   alpha=0.55, linewidths=0, zorder=3)
        # Mean carries a surface ring so it stays legible where seeds pile up on it.
        ax.scatter([v.mean()], [y], s=132, color=color, edgecolors="#fcfcfb",
                   linewidths=2.0, zorder=4)
        ax.text(v.mean(), y + 0.33, f"{v.mean():.2f}", ha="center", va="bottom",
                fontsize=9, color=MUTED, zorder=5)

    frozen = np.mean(by_arm[PRETRAIN]) * 1e3 if PRETRAIN in by_arm else None
    if frozen is not None:
        ax.axvline(frozen, color=STAGE_COLOR["frozen"], lw=1.2, ls=(0, (4, 3)),
                   alpha=0.55, zorder=1)
        # Rotated alongside the line, mid-height: the top of the line is where pretrain's own
        # mean label sits, and the bottom is the legend.
        ax.text(frozen * 1.04, (len(arms) - 1) / 2.0, "do nothing", rotation=90,
                ha="left", va="center", fontsize=8.5, color=STAGE_COLOR["frozen"])

    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a[1] for a in reversed(arms)], fontsize=10, color=INK)
    ax.set_xlabel(r"one-step MSE on new-regime validation  ($\times 10^{-3}$)",
                  fontsize=10, color=MUTED)
    # Log x: pretrain is ~4.8x the adapted arms, which on a linear axis squeezes all eight of
    # them into the left fifth and hides the differences the sweep exists to show. MSE is
    # ratio-scale and every comparison here ("closes half the gap") is a ratio, so log is the
    # honest axis rather than a cosmetic one.
    ax.set_xscale("log")
    ax.set_xlim(0.85, 5.9)
    ax.set_xticks([1, 1.5, 2, 3, 4, 5])
    ax.set_xticklabels(["1", "1.5", "2", "3", "4", "5"])
    ax.minorticks_off()
    ax.set_ylim(-0.75, len(arms) - 0.15)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=9)

    handles = [plt.Line2D([], [], marker="o", ls="", markersize=8, color=c,
                          markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=n)
               for n, c in (("frozen pre-shift", STAGE_COLOR["frozen"]),
                            ("cold (from scratch)", STAGE_COLOR["cold"]),
                            ("warm (from pretrain)", STAGE_COLOR["warm"]))]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9,
              labelcolor=MUTED, handletextpad=0.4, borderaxespad=1.2)

    ax.set_title("Lower is better on the post-shift regime", fontsize=13, color=INK,
                 loc="left", pad=32)
    ax.text(0.0, 1.03, f"F 8.0 \u2192 10.0, 6000 steps, one dot per seed{title_note}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=MUTED)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp-dir", default="results/shift")
    ap.add_argument("--out", default="figures/shift_new_regime_mse.png")
    args = ap.parse_args()
    by_arm = load(args.exp_dir)
    n = {len(v) for v in by_arm.values()}
    note = "" if n == {5} else f"  — INCOMPLETE: {sorted(n)} seeds per arm"
    plot(by_arm, args.out, note)


if __name__ == "__main__":
    main()
