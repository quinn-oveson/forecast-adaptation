import torch

DORMANT_TAU = 0.025  # Sokar et al. dormancy threshold on normalized mean activation.


@torch.no_grad()
def features(model, x):
    # Penultimate activations: everything up to but excluding the output head.
    if x.dim() == 2:
        x = x.unsqueeze(1)
    return model.body(x)


@torch.no_grad()
def effective_rank(model, x, eps=1e-12):
    # Entropy-based effective rank of the feature matrix (channels x samples*sites).
    h = features(model, x)
    # SVD on CPU: aten::_linalg_svd is unimplemented on MPS, and this runs once per cycle.
    m = h.permute(1, 0, 2).reshape(h.shape[1], -1).float().cpu()
    m = m - m.mean(dim=1, keepdim=True)
    s = torch.linalg.svdvals(m)
    p = s / (s.sum() + eps)
    p = p[p > eps]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def dormant_fraction(model, x, tau=DORMANT_TAU):
    # Fraction of channels whose mean activation is negligible relative to the layer mean.
    h = features(model, x).abs().mean(dim=(0, 2))
    score = h / (h.mean() + 1e-12)
    return float((score < tau).float().mean())


@torch.no_grad()
def layer_norms(model):
    # Per-layer weight L2 norms, keyed by parameter name.
    return {n: float(p.detach().norm()) for n, p in model.named_parameters() if p.dim() > 1}


def norm_ratios(current, initial):
    # ||W|| / ||W_init|| per layer: determines what a given shrink-and-perturb lambda does.
    return {k: current[k] / initial[k] if initial.get(k) else float("nan") for k in current}


@torch.no_grad()
def collect(model, x, init_norms=None):
    # One flat dict of every diagnostic, ready to widen a CSV row.
    cur = layer_norms(model)
    out = {"effective_rank": effective_rank(model, x),
           "dormant_frac": dormant_fraction(model, x),
           "weight_norm": float(sum(v ** 2 for v in cur.values()) ** 0.5)}
    for n, v in cur.items():
        out[f"wnorm::{n}"] = v
    if init_norms:
        for n, r in norm_ratios(cur, init_norms).items():
            out[f"wratio::{n}"] = r
    return out


def diagnostic_columns(model, init_norms=True):
    # Column names in a fixed order, so the CSV header is stable before any training runs.
    names = [n for n, p in model.named_parameters() if p.dim() > 1]
    cols = ["effective_rank", "dormant_frac", "weight_norm"]
    cols += [f"wnorm::{n}" for n in names]
    if init_norms:
        cols += [f"wratio::{n}" for n in names]
    return cols
