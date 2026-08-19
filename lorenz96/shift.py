import torch

from .stream import VAL_SEED_OFFSET, Split, _load

# One trajectory seed per pool. Offsets are spaced far wider than any plausible seed grid so
# that seed s of one pool can never land on seed s' of another; the constructor checks it
# rather than assuming it, because S5 records that the inherited leakage assertion compares
# trajectories that are independent by construction and so cannot fail.
OFFSETS = dict(old_train=0, new_train=11_000, val_old=VAL_SEED_OFFSET, val_new=13_000)


class ShiftStream:
    # A large pre-shift pool and a small post-shift one, with held-out validation from separate
    # trajectories in both regimes. Unlike CycleStream this is deliberately asymmetric: S7 notes
    # that equal chunks do not represent years of old analysis against months of new.
    def __init__(self, seed, noise, history, F_old, F_new, n_old, n_new, n_val):
        self.seeds = {name: seed + off for name, off in OFFSETS.items()}
        if len(set(self.seeds.values())) != len(self.seeds):
            raise ValueError(f"pool seeds collide at base seed {seed}: {self.seeds}")
        self.F_old, self.F_new = F_old, F_new
        self.n_old, self.n_new = n_old, n_new

        self.old = _load(n_old, self.seeds["old_train"], noise, history, F_old)
        self.new = _load(n_new, self.seeds["new_train"], noise, history, F_new)
        # Selection uses the new regime -- that is what the model is being adapted to. The old
        # regime is validated too so that forgetting is measured rather than inferred.
        self.val_new = _load(n_val, self.seeds["val_new"], noise, history, F_new)
        self.val_old = _load(n_val, self.seeds["val_old"], noise, history, F_old)

    def pooled(self):
        # Old first, then new: indices below n_old are old, at or above it are new. MixSampler
        # depends on that layout. Cached: a driver sweeping arms asks for it once per arm.
        if getattr(self, "_pooled", None) is None:
            self._pooled = Split(*(torch.cat(parts, dim=0) for parts in zip(self.old, self.new)))
        return self._pooled


def resolve_ratio(mix_ratio, n_old, n_new):
    # "natural" is the pooled proportion, which makes train-on-everything a point on the same
    # sweep as the replay arms instead of a separately implemented arm.
    if mix_ratio == "natural":
        return n_new / (n_old + n_new)
    return float(mix_ratio)


class MixSampler:
    # E4 by per-batch composition: every batch is `ratio` new and the rest old, drawn with
    # replacement from the full pools. Steps, batch size and samples-seen are then identical at
    # every ratio, so the ratio is the only quantity that varies across the sweep, and none of
    # the scarce new data is discarded at low ratios.
    def __init__(self, n_old, n_new, ratio, batch_size, generator):
        self.n_old, self.n_new, self.batch_size = n_old, n_new, batch_size
        self.requested = ratio
        self.n_new_per = int(round(ratio * batch_size))
        self.n_old_per = batch_size - self.n_new_per
        if self.n_new_per and not n_new:
            raise ValueError(f"ratio {ratio} needs new-regime data but n_new={n_new}")
        if self.n_old_per and not n_old:
            raise ValueError(f"ratio {ratio} needs old-regime data but n_old={n_old}")
        # Rounding to whole samples means the realized ratio is not always the requested one.
        self.realized = self.n_new_per / batch_size
        self.generator = generator

    def __iter__(self):
        while True:
            parts = []
            if self.n_old_per:
                parts.append(torch.randint(0, self.n_old, (self.n_old_per,),
                                           generator=self.generator))
            if self.n_new_per:
                parts.append(self.n_old + torch.randint(0, self.n_new, (self.n_new_per,),
                                                        generator=self.generator))
            yield torch.cat(parts)

    def log(self):
        return dict(mix_ratio_requested=self.requested, mix_ratio_realized=self.realized,
                    mix_new_per_batch=self.n_new_per, mix_old_per_batch=self.n_old_per)
