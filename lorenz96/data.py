from typing import NamedTuple

import numpy as np
import torch

from .system import DT, f_schedule, initial_state, integrate

REFERENCE_F = 8.0
SPINUP = 5_000
STRIDE = 5  # Model timestep 0.05 MTU (6 h); integrator stays at DT for accuracy.

_traj_cache = {}
_norm_cache = {}


class Batch(NamedTuple):
    x: torch.Tensor        # model input, noisy when noise > 0
    y: torch.Tensor        # model target, noisy when noise > 0
    x_clean: torch.Tensor
    y_clean: torch.Tensor


def load_trajectory(F=REFERENCE_F, n_steps=100_000, seed=0, dt=DT, spinup=SPINUP):
    key = (float(F) if np.isscalar(F) else None, n_steps, seed, dt, spinup)
    if key[0] is None:
        return integrate(initial_state(REFERENCE_F, seed), F, n_steps, dt, spinup)
    if key not in _traj_cache:
        x0 = initial_state(F, seed)
        _traj_cache[key] = integrate(x0, F, n_steps, dt, spinup)
    return _traj_cache[key]


def normalizer(n_steps=100_000, seed=0):
    # Frozen at REFERENCE_F: Var(X) grows with F, so per-regime stats would fake a shift effect.
    key = (n_steps, seed)
    if key not in _norm_cache:
        traj = load_trajectory(REFERENCE_F, n_steps, seed)
        _norm_cache[key] = (float(traj.mean()), float(traj.std()))
    return _norm_cache[key]


def _to_pairs(traj, stride, noise, seed):
    # Noise is applied once to the trajectory so consecutive observation errors stay correlated.
    mean, std = normalizer()
    clean = (traj - mean) / std

    if noise > 0:
        rng = np.random.default_rng(seed + 9973)
        obs = clean + noise * rng.standard_normal(clean.shape)
    else:
        obs = clean

    sub, sub_clean = obs[::stride], clean[::stride]
    x, y = sub[:-1], sub[1:] - sub[:-1]
    xc, yc = sub_clean[:-1], sub_clean[1:] - sub_clean[:-1]

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
    return Batch(t(x), t(y), t(xc), t(yc))


def make_dataset(F=REFERENCE_F, n=10_000, seed=0, stride=STRIDE, noise=0.0, dt=DT,
                 spinup=SPINUP):
    # Targets are tendencies over one model step, not raw next states.
    traj = load_trajectory(F, n * stride + 1, seed, dt, spinup)
    return _to_pairs(traj, stride, noise, seed)


# The subset draw gets its own band so it stays independent of the trajectory and noise
# streams derived from the same base seed.
SUBSET_SEED_OFFSET = 3_000


def load_subset(n, seed, F=REFERENCE_F, pool=20_000, **kwargs):
    # The pool follows `seed`, so one seed is one complete replication: trajectory, noise draw
    # and subset all move together. Pinning the pool at seed=0 -- what this did before -- left
    # only the subset selection varying, which makes an across-seed error bar a measure of
    # subset choice and weight init alone and understates the true spread. ShiftStream already
    # derives its pools from the run seed; this matches it.
    full = make_dataset(F, pool, seed=seed, **kwargs)
    g = torch.Generator().manual_seed(seed + SUBSET_SEED_OFFSET)
    idx = torch.randperm(full.x.shape[0], generator=g)[:n]
    return Batch(*(t[idx] for t in full))


def make_shifted_dataset(kind, n, F0=8.0, F1=16.0, width=0, n_cycles=2, seed=0,
                         stride=STRIDE, noise=0.0, dt=DT, spinup=SPINUP):
    # Non-stationary stream: F varies across the trajectory per f_schedule.
    raw = n * stride + 1
    F_seq = np.concatenate(
        [np.full(spinup, F0), f_schedule(kind, raw, F0, F1, width, n_cycles)])
    traj = integrate(initial_state(F0, seed), F_seq, raw, dt, spinup)
    batch = _to_pairs(traj, stride, noise, seed)
    return batch, F_seq[spinup::stride][:batch.x.shape[0]]


def rollout_truth(F=REFERENCE_F, n_steps=1_000, seed=0, stride=STRIDE, dt=DT, spinup=SPINUP):
    # Clean subsampled trajectory in normalized units, for verification only.
    traj = load_trajectory(F, n_steps * stride + 1, seed, dt, spinup)
    mean, std = normalizer()
    return torch.from_numpy(((traj - mean) / std)[::stride]).float()
