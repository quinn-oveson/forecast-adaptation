import numpy as np
import torch

from .data import STRIDE, rollout_truth
from .system import DT

# Measured by explore.py; see TRANSFER.md. Forecast horizons are normalized by these.
# lambda1 is concave in F, so linear interpolation between knots UNDERESTIMATES it and
# therefore overestimates any horizon expressed in Lyapunov times. Exact at the measured
# forcings, which includes both regimes of the shift experiment (F=8 and F=10).
LAMBDA1 = {3.0: 0.0, 4.0: 0.0, 5.0: 0.523, 6.0: 0.964, 8.0: 1.671,
           10.0: 2.278, 12.0: 2.873, 16.0: 3.849}

# 20 model steps = 1 MTU ~= 2 decorrelation times, so rollout starts are near-independent.
START_GAP = 20

# Rollout trajectories must never reuse a training pool seed. Taken: 0 old-regime train,
# 3_000 subset draw, 5_000 val/val_old, 7_000 test, 9_973 noise, 11_000 new-regime train,
# 13_000 val_new. Evaluation gets its own band above all of them.
EVAL_SEED_OFFSET = 17_000


def lambda1(F):
    grid = np.array(sorted(LAMBDA1))
    return float(np.interp(F, grid, [LAMBDA1[f] for f in grid]))


@torch.no_grad()
def rollout(model, x0, n_steps, sample=False):
    # Autoregressive: the model predicts a tendency that is added back to the state.
    model.eval()
    history = (x0 if x0.dim() == 3 else x0.unsqueeze(1)).contiguous()
    state = history[:, -1]
    out = torch.empty(x0.shape[0], n_steps, state.shape[-1], device=state.device)

    for t in range(n_steps):
        pred = model(history if history.shape[1] > 1 else state)
        if isinstance(pred, tuple):
            mean, log_var = pred
            pred = mean + torch.randn_like(mean) * (0.5 * log_var).exp() if sample else mean
        state = state + pred
        out[:, t] = state
        history = torch.cat([history[:, 1:], state.unsqueeze(1)], dim=1)
    return out


def nrmse_curve(pred, truth, per_regime=False):
    # `truth` arrives in frozen-normalized units (data.normalizer, pinned at REFERENCE_F), so
    # the reference climatology is 1.0 by construction: 1.0 on this curve is the error of two
    # unrelated states drawn from the REFERENCE attractor, whatever F is being evaluated.
    #
    # Taking truth.std() here -- what this did before -- rescales by the evaluation regime's
    # own spread and cancels exactly the normalization data.py freezes. Var(X) grows with F, so
    # it deflates every error at F > 8 (17% at F=10, 28% at F=12) and flatters the post-shift
    # regime. A frozen denominator is what makes two forcings comparable at all.
    #
    # per_regime=True restores the skill-score reading (error against the LOCAL climatology).
    # That is a legitimate but different question, and its numbers are not comparable across F.
    clim = float(truth.reshape(-1, truth.shape[-1]).std()) if per_regime else 1.0
    err = ((pred - truth) ** 2).mean(dim=(0, 2)).sqrt()
    return err / (np.sqrt(2.0) * clim)


def valid_prediction_time(pred, truth, threshold=0.3, F=8.0, stride=STRIDE, dt=DT,
                          per_regime=False):
    # Lead time at which the normalized error curve crosses `threshold`, in Lyapunov times.
    #
    # The crossing is linearly interpolated between the two bracketing eval points. Without
    # that the answer is quantized to stride * dt = 0.05 MTU, and any arm difference below
    # ~0.1 MTU is unresolvable -- which is the scale the replay sweep is trying to resolve.
    # Interpolation is free and, unlike shrinking STRIDE, does not invalidate a single trained
    # checkpoint. `steps` is therefore fractional.
    #
    # `censored` marks the two cases where the number is a bound, not a measurement:
    #   "left"   the curve starts above threshold -- never skilful. 0.0 is exact but says
    #            nothing about HOW bad; averaging it against real horizons is meaningless.
    #   "right"  the curve never crosses -- the horizon is a LOWER bound (>= rollout length),
    #            so a mean mixing these with real crossings is biased low.
    # Both were previously indistinguishable from measured values in the returned dict.
    curve = nrmse_curve(pred, truth, per_regime=per_regime)
    over = (curve > threshold).nonzero()

    if not len(over):
        steps, censored = float(len(curve)), "right"
    elif int(over[0]) == 0:
        steps, censored = 0.0, "left"
    else:
        i = int(over[0])
        lo, hi = float(curve[i - 1]), float(curve[i])
        steps = (i - 1) + ((threshold - lo) / (hi - lo) if hi > lo else 1.0)
        censored = None

    mtu = steps * stride * dt
    lam = lambda1(F)
    return dict(steps=steps, mtu=mtu, lyapunov_times=mtu * lam if lam > 0 else float("nan"),
                censored=censored, curve=curve)


@torch.no_grad()
def evaluate(model, F=8.0, n_steps=200, n_init=64, seed=EVAL_SEED_OFFSET, history=1,
             sample=False, threshold=0.3, stride=STRIDE, init_noise=0.0, start_gap=START_GAP,
             per_regime=False):
    # Initialize from noisy observations (an analysis) but score against clean truth.
    #
    # seed defaults into the evaluation band so a rollout can never land on a training pool.
    # The old default of 0 IS the old-regime training seed, which is how the predecessor's
    # RESULTS.md came out in-sample. An evaluator scoring a run should still pass
    # run_seed + EVAL_SEED_OFFSET, so the eval trajectory varies with the run it is scoring.
    dev = next(model.parameters(), torch.zeros(1)).device
    # Starts are spaced >=1 decorrelation time apart; adjacent starts would be ~1 sample.
    span = (n_init - 1) * start_gap
    truth = rollout_truth(F, span + n_steps + history + 2, seed=seed, stride=stride)
    g = torch.Generator().manual_seed(seed)
    slack = max(1, len(truth) - n_steps - history - 1 - span)
    off = int(torch.randint(0, slack, (1,), generator=g))
    starts = off + torch.arange(n_init) * start_gap

    obs = truth if init_noise == 0 else truth + init_noise * torch.randn(
        truth.shape, generator=g)
    x0 = torch.stack([obs[s:s + history] for s in starts]).contiguous().to(dev)
    target = torch.stack(
        [truth[s + history:s + history + n_steps] for s in starts]).contiguous().to(dev)

    pred = rollout(model, x0 if history > 1 else x0.squeeze(1), n_steps, sample=sample)
    res = valid_prediction_time(pred, target, threshold, F, stride, per_regime=per_regime)
    res["stable"] = bool(torch.isfinite(pred).all() and pred.abs().max() < 50)
    res["pred_std"] = float(pred[:, -1].std())
    res["truth_std"] = float(target[:, -1].std())
    return res
