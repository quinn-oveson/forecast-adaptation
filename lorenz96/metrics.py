import numpy as np
import torch

from .data import STRIDE, rollout_truth
from .system import DT

# Measured by explore.py; see TRANSFER.md. Forecast horizons are normalized by these.
LAMBDA1 = {3.0: 0.0, 4.0: 0.0, 5.0: 0.523, 6.0: 0.964, 8.0: 1.671,
           10.0: 2.278, 12.0: 2.873, 16.0: 3.849}

# 20 model steps = 1 MTU ~= 2 decorrelation times, so rollout starts are near-independent.
START_GAP = 20


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


def nrmse_curve(pred, truth):
    # Normalized so 1.0 is the error of two unrelated states drawn from the attractor.
    clim = truth.reshape(-1, truth.shape[-1]).std()
    err = ((pred - truth) ** 2).mean(dim=(0, 2)).sqrt()
    return err / (np.sqrt(2.0) * clim)


def valid_prediction_time(pred, truth, threshold=0.3, F=8.0, stride=STRIDE, dt=DT):
    # Steps until normalized error crosses the threshold, expressed in Lyapunov times.
    curve = nrmse_curve(pred, truth)
    over = (curve > threshold).nonzero()
    steps = int(over[0]) if len(over) else len(curve)
    mtu = steps * stride * dt
    lam = lambda1(F)
    return dict(steps=steps, mtu=mtu, lyapunov_times=mtu * lam if lam > 0 else float("nan"),
                curve=curve)


@torch.no_grad()
def evaluate(model, F=8.0, n_steps=200, n_init=64, seed=0, history=1, sample=False,
             threshold=0.3, stride=STRIDE, init_noise=0.0, start_gap=START_GAP):
    # Initialize from noisy observations (an analysis) but score against clean truth.
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
    res = valid_prediction_time(pred, target, threshold, F, stride)
    res["stable"] = bool(torch.isfinite(pred).all() and pred.abs().max() < 50)
    res["pred_std"] = float(pred[:, -1].std())
    res["truth_std"] = float(target[:, -1].std())
    return res
