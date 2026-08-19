from typing import NamedTuple

import torch

from .data import REFERENCE_F, make_dataset

# Offsets keep train/val/test on independent trajectories so no window can straddle a split.
VAL_SEED_OFFSET = 5_000
TEST_SEED_OFFSET = 7_000


class Split(NamedTuple):
    x: torch.Tensor        # (n, history, K) noisy observation windows
    x_clean: torch.Tensor  # (n, history, K) true state, evaluation only
    y: torch.Tensor        # (n, K) noisy tendency target
    y_clean: torch.Tensor  # (n, K) true tendency, evaluation only

    # x_clean is carried so validation can score the predicted NEXT STATE, x[:, -1] + pred,
    # against the true next state x_clean[:, -1] + y_clean. Scoring the tendency against
    # y_clean instead penalises the model for the -eps_t term it must emit: given a noisy
    # input, the optimal tendency is E[c_t+1 | x] - x, which contains it by construction.


def derive_seeds(seed, cycle):
    # Every random stream is a deterministic function of the base seed, and is recorded.
    return dict(seed=seed,
                data_seed=seed,
                noise_seed=seed + 9973,
                init_seed=101 + seed * 1000 + cycle,
                shuffle_seed=500_003 + seed * 100_003 + cycle)


def _stack_history(b, history):
    # Sliding windows of past observations; targets align to the last frame of each window.
    if history == 1:
        return Split(b.x.unsqueeze(1), b.x_clean.unsqueeze(1), b.y, b.y_clean)
    stack = lambda a: torch.stack([a[i:len(a) - history + 1 + i] for i in range(history)], dim=1)
    return Split(stack(b.x), stack(b.x_clean), b.y[history - 1:], b.y_clean[history - 1:])


def _load(n, seed, noise, history, F):
    # n + history - 1 pairs yields exactly n stacked windows.
    return _stack_history(make_dataset(F, n + history - 1, seed=seed, noise=noise), history)


class CycleStream:
    # Data arriving in equal chunks, with fixed held-out val/test from separate trajectories.
    def __init__(self, seed=0, noise=0.05, n_cycles=10, chunk=2_000, n_val=4_000,
                 history=4, F=REFERENCE_F, n_test=None):
        self.seed, self.noise, self.n_cycles, self.chunk = seed, noise, n_cycles, chunk
        self.history, self.F = history, F

        # Chunk seeds are `seed * 31 + c`, so c must stay inside that stride: at c >= 31, chunk
        # c of seed s IS chunk c - 31 of seed s + 1, and two runs silently train on identical
        # data while reporting as independent seeds. Refuse rather than alias.
        if n_cycles > 31:
            raise ValueError(
                f"n_cycles={n_cycles} exceeds the 31-wide stride of `seed * 31 + c`: chunk "
                "seeds alias across base seeds. Widen the multiplier in _chunk_seed and "
                "regenerate every affected run before going past 31 cycles.")

        # Splits must sit on separate trajectories, and that is enforced by construction here.
        self.split_seeds = dict(val=seed + VAL_SEED_OFFSET, test=seed + TEST_SEED_OFFSET,
                                **{f"chunk{c}": self._chunk_seed(c) for c in range(n_cycles)})
        counts = list(self.split_seeds.values())
        dupes = sorted(k for k, v in self.split_seeds.items() if counts.count(v) > 1)
        if dupes:
            raise ValueError(f"split seeds collide at base seed {seed}: {dupes}")

        self.val = _load(n_val, seed + VAL_SEED_OFFSET, noise, history, F)
        # Shares the rollout initial conditions' trajectory; only ever reported, never selected on.
        self.test_seed = seed + TEST_SEED_OFFSET
        self.test = _load(n_test or n_val, self.test_seed, noise, history, F)
        self._chunks = [_load(chunk, self._chunk_seed(c), noise, history, F)
                        for c in range(n_cycles)]

    def _chunk_seed(self, c):
        return self.seed * 31 + c

    def cycle_data(self, cycle, data_all):
        # data_all=True gives everything seen so far; False gives only the newest chunk.
        use = self._chunks[:cycle + 1] if data_all else self._chunks[cycle:cycle + 1]
        return Split(*(torch.cat(t, dim=0) for t in zip(*use)))

    def seeds(self, cycle):
        return derive_seeds(self.seed, cycle)
