"""Shared helpers for the plotting scripts."""
from pathlib import Path


def figure_path(exp_dir, name, override=None):
    """figures/<experiment>/<name>.png -- one folder per experiment directory.

    Figures are namespaced by the basename of --exp-dir so that plotting a second experiment
    cannot silently overwrite the first one's output. Filenames stay stable across experiments,
    which is what makes two runs comparable side by side:

        figures/shift/new_regime_mse.png
        figures/shift_n500/new_regime_mse.png

    --out still overrides completely, for a one-off path.
    """
    if override:
        return Path(override)
    return Path("figures") / Path(exp_dir).resolve().name / f"{name}.png"
