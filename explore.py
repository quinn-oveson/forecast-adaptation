import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lorenz96 import data as D
from lorenz96 import system as S

FIGDIR = "figures"
INK, MUTED = "#0b0b0b", "#52514e"
GRID = "#dcdcd8"
SERIES = ["#2a78d6", "#eb6834"]
F_VALUES = [4.0, 8.0, 12.0, 16.0]
F_COLORS = [plt.cm.Blues(t) for t in (0.35, 0.55, 0.75, 0.95)]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "axes.titlesize": 11,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "legend.frameon": False,
})


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def fig_hovmoller():
    # Diverging map centered at zero shows the travelling waves and their polarity.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, F in zip(axes, (8.0, 16.0)):
        traj = D.load_trajectory(F, 1500)
        lim = np.abs(traj).max()
        im = ax.imshow(traj.T, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-lim, vmax=lim,
                       extent=[0, 1500 * S.DT, 0, S.K], interpolation="nearest")
        ax.set_title(f"F = {F:g}", color=INK)
        ax.set_xlabel("time (MTU)")
        ax.set_ylabel("site k" if F == 8.0 else "")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="$X_k$", pad=0.02)
    fig.suptitle("Lorenz 96 travelling waves: stronger forcing sharpens and speeds the waves",
                 color=INK, fontsize=11)
    fig.savefig(f"{FIGDIR}/fig1_hovmoller.png", dpi=160)
    plt.close(fig)


def fig_attractor_stats():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)

    for F, c in zip(F_VALUES, F_COLORS):
        traj = D.load_trajectory(F, 60_000)
        axes[0].hist(traj.ravel(), bins=120, density=True, histtype="step",
                     color=c, lw=1.6, label=f"F = {F:g}")

        lags = np.arange(0, 200)
        x = traj[:, 0] - traj[:, 0].mean()
        ac = np.correlate(x, x, "full")[len(x) - 1:len(x) - 1 + len(lags)]
        axes[2].plot(lags * S.DT, ac / ac[0], color=c, lw=1.6, label=f"F = {F:g}")

    axes[0].set_xlabel("$X_k$")
    axes[0].set_ylabel("density")
    axes[0].set_title("Marginal distribution shifts with F")
    axes[0].legend()

    F_dense = np.arange(3.0, 17.1, 1.0)
    stds = [D.load_trajectory(F, 20_000).std() for F in F_dense]
    axes[1].plot(F_dense, stds, color=SERIES[0], lw=2, marker="o", ms=4)
    axes[1].set_xlabel("forcing F")
    axes[1].set_ylabel("std of $X_k$")
    axes[1].set_title("Variance grows with F\n(why normalization must be frozen)")

    axes[2].axhline(0, color=GRID, lw=1)
    axes[2].set_xlabel("lag (MTU)")
    axes[2].set_ylabel("autocorrelation")
    axes[2].set_title("Decorrelation is faster at higher F")
    axes[2].legend()

    for ax in axes:
        _style(ax)
    fig.savefig(f"{FIGDIR}/fig2_attractor_stats.png", dpi=160)
    plt.close(fig)
    return dict(zip(F_dense, stds))


def fig_lyapunov():
    F_sweep = np.array([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0])
    les = np.array([S.lyapunov_exponent(F, n_steps=20_000, spinup=4_000) for F in F_sweep])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    axes[0].axhline(0, color=GRID, lw=1)
    axes[0].plot(F_sweep, les, color=SERIES[0], lw=2, marker="o", ms=5)
    i8 = int(np.where(F_sweep == 8.0)[0][0])
    axes[0].annotate(f"F=8: $\\lambda_1$={les[i8]:.2f}\n(lit. $\\approx$1.7)",
                     xy=(8.0, les[i8]), xytext=(8.6, les[i8] - 1.15), color=MUTED,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    axes[0].set_xlabel("forcing F")
    axes[0].set_ylabel("$\\lambda_1$ (1/MTU)")
    axes[0].set_title("Leading Lyapunov exponent")

    # Only the chaotic regimes have a meaningful horizon; 1/lambda diverges as lambda -> 0.
    chaotic = les > 0.1
    axes[1].plot(F_sweep[chaotic], 1.0 / les[chaotic], color=SERIES[0], lw=2, marker="o", ms=5)
    axes[1].set_xlabel("forcing F")
    axes[1].set_ylabel("Lyapunov time (MTU)")
    axes[1].set_title("Predictability horizon shrinks as F grows")
    axes[1].set_xlim(*axes[0].get_xlim())
    axes[1].axvspan(axes[0].get_xlim()[0], 4.5, color=GRID, alpha=0.5, lw=0)
    axes[1].text(3.6, 1.55, "non-chaotic", color=MUTED, fontsize=8, rotation=90, va="center")

    for ax in axes:
        _style(ax)
    fig.savefig(f"{FIGDIR}/fig3_lyapunov.png", dpi=160)
    plt.close(fig)
    return dict(zip(F_sweep, les))


def fig_divergence(le8, n_pairs=40):
    # Average over many pairs: a single pair shows local Lyapunov bursts, not the global rate.
    rng = np.random.default_rng(0)
    base = D.load_trajectory(8.0, 60_000)
    n = 1200
    seps = np.empty((n_pairs, n))

    for p in range(n_pairs):
        a = base[rng.integers(0, len(base))].copy()
        b = a + 1e-8 * rng.standard_normal(S.K)
        for t in range(n):
            seps[p, t] = np.linalg.norm(a - b)
            a, b = S.rk4_step(a, 8.0), S.rk4_step(b, 8.0)

    geo_mean = np.exp(np.log(seps).mean(axis=0))
    time = np.arange(n) * S.DT
    saturation = np.sqrt(2 * S.K) * base.std()

    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    for p in range(n_pairs):
        ax.semilogy(time, seps[p], color=SERIES[0], lw=0.4, alpha=0.15)
    ax.semilogy(time, geo_mean, color=SERIES[0], lw=2,
                label=f"mean separation ({n_pairs} pairs)")
    ax.semilogy(time, geo_mean[0] * np.exp(le8 * time), color=SERIES[1], lw=1.6, ls="--",
                label=f"$e^{{\\lambda_1 t}}$, $\\lambda_1$={le8:.2f}")
    ax.axhline(saturation, color=MUTED, lw=1, ls=":")
    ax.text(0.3, saturation * 1.4, "saturation (climatological error)",
            color=MUTED, fontsize=8)
    ax.set_ylim(1e-9, saturation * 12)
    ax.set_xlabel("time (MTU)")
    ax.set_ylabel("$\\|X - X'\\|$")
    ax.set_title("A perfect model still diverges\nRMSE alone is the wrong metric")
    ax.legend(loc="lower right")
    _style(ax)
    fig.savefig(f"{FIGDIR}/fig4_divergence.png", dpi=160)
    plt.close(fig)


def fig_schedules():
    n = 12_000
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True, constrained_layout=True)

    for (kind, width, label), c in zip(
            [("step", 0, "step (abrupt)"), ("ramp", 6_000, "ramp (gradual)")], SERIES):
        X, _, F_seq = D.make_shifted_dataset(kind, n, 8.0, 16.0, width=width)
        axes[0].plot(np.arange(n) * S.DT, F_seq, color=c, lw=2, label=label)

        w = 400
        rolling = np.array([X[i:i + w].std() for i in range(0, n - w, 50)])
        axes[1].plot(np.arange(len(rolling)) * 50 * S.DT, rolling, color=c, lw=1.6, label=label)

    axes[0].set_ylabel("forcing F")
    axes[0].set_title("The abruptness knob: same endpoints, different transition")
    axes[0].legend()
    axes[1].set_xlabel("time (MTU)")
    axes[1].set_ylabel("rolling std of $X$")
    axes[1].set_title("Resulting non-stationarity in the data")
    axes[1].legend()

    for ax in axes:
        _style(ax)
    fig.savefig(f"{FIGDIR}/fig5_schedules.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    fig_hovmoller()
    stds = fig_attractor_stats()
    les = fig_lyapunov()
    fig_divergence(les[8.0])
    fig_schedules()

    print(f"{'F':>6} {'std':>8} {'lambda_1':>10} {'Lyap time':>10}")
    for F in sorted(les):
        lt = 1.0 / les[F] if les[F] > 1e-3 else float("inf")
        print(f"{F:6.1f} {stds.get(F, float('nan')):8.3f} {les[F]:10.4f} {lt:10.3f}")
    print(f"\nwrote 5 figures to {FIGDIR}/")
