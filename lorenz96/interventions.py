from dataclasses import dataclass
from typing import Any, Optional

import torch

# Loss-of-plasticity interventions. The three known methods do not share a call site, so the
# protocol exposes all three and each method fills in the one it needs:
#
#   on_run_start   one-shot weight transform before the first step   Shrink & Perturb, DASH
#   grad_pass      a gradient pass over the arriving data first      DASH
#   on_step_end    per-step unit replacement and optimizer surgery   Continual Backprop
#
# A method that lands later touches only this file.


@dataclass
class RunStart:
    model: Any
    optimizer: Any
    fresh_state: Any                 # state_dict of a freshly initialized identical model
    generator: Any                   # torch.Generator, so perturbations are seeded
    grad_ema: Optional[Any] = None   # name -> tensor; supplied only if requires_grad_pass


class Intervention:
    kind = "none"

    def __init__(self, params=None):
        self.params = dict(params or {})
        self._log = {}

    def requires_grad_pass(self):
        return False

    def on_run_start(self, ctx):
        return None

    def on_step_end(self, model, optimizer, step):
        return None

    def log(self):
        return dict(self._log)

    def _require(self, *names):
        missing = [n for n in names if n not in self.params]
        if missing:
            raise ValueError(f"intervention {self.kind!r} needs params {missing}")
        extra = [n for n in self.params if n not in names]
        if extra:
            raise ValueError(f"intervention {self.kind!r} got unknown params {extra}")
        return [self.params[n] for n in names]


class NoOp(Intervention):
    kind = "none"

    def __init__(self, params=None):
        super().__init__(params)
        self._require()


class ShrinkPerturb(Intervention):
    # Ash & Adams: theta <- shrink * theta + noise_scale * theta_fresh, where theta_fresh is a
    # draw from the initialization distribution rather than isotropic noise, so each tensor is
    # perturbed at its own layer's scale. Applied to every parameter, biases included.
    kind = "snp"

    def __init__(self, params=None):
        super().__init__(params)
        self.shrink, self.noise_scale = self._require("shrink", "noise_scale")

    @torch.no_grad()
    def on_run_start(self, ctx):
        before = _global_norm(ctx.model)
        for name, p in ctx.model.named_parameters():
            fresh = ctx.fresh_state[name].to(p.device, p.dtype)
            p.mul_(self.shrink).add_(fresh, alpha=self.noise_scale)
        after = _global_norm(ctx.model)
        self._log = dict(snp_norm_before=before, snp_norm_after=after,
                         snp_norm_ratio=after / before if before else float("nan"))


class Dash(Intervention):
    # Shin et al. 2024 (NeurIPS), Algorithm 1: per-neuron incoming weight vector theta is scaled
    # by max{lambda, cos_sim(-G, theta)} where G is an EMA over per-chunk loss gradients,
    # G <- (1-alpha) G + alpha * grad(loss on chunk i), accumulated over chunks 1..j.
    kind = "dash"

    def __init__(self, params=None):
        super().__init__(params)
        self.lam, self.alpha = self._require("lam", "alpha")

    def requires_grad_pass(self):
        return True

    def on_run_start(self, ctx):
        raise NotImplementedError(
            "DASH is a validated seam, not an implementation. Two pieces are missing:\n"
            "  1. The per-neuron shrink itself: group each conv weight by (out, in) filter, "
            "flatten the kernel to a vector theta, and scale by max{lam, cos_sim(-G, theta)}. "
            "Paper uses lam=0.05.\n"
            "  2. G is an EMA over the gradients of chunks 1..j, but train.py is a single-run "
            "trainer that only ever sees this run's chunk. G therefore has to be carried "
            "between runs -- either persisted alongside the checkpoint or recomputed from all "
            "prior chunks by the driver -- and that carrier does not exist yet. With alpha=1 "
            "the EMA collapses to this chunk's gradient alone, which is the only case the "
            "current single-run interface can express.\n"
            "ctx.grad_ema already carries this run's chunk gradient (requires_grad_pass=True).")


class ContinualBackprop(Intervention):
    # Dohare, Sutton, Mahmood et al., Nature 2024. Per-step generate-and-test: a fraction of
    # mature, low-utility units is reinitialized continually rather than only at step 0.
    kind = "cbp"

    def __init__(self, params=None):
        super().__init__(params)
        self.replacement_rate, self.decay_rate, self.maturity_threshold, self.utility = (
            self._require("replacement_rate", "decay_rate", "maturity_threshold", "utility"))

    def on_step_end(self, model, optimizer, step):
        raise NotImplementedError(
            "Continual Backprop is a validated seam, not an implementation. Four pieces are "
            "missing, all per conv layer:\n"
            "  1. Contribution utility |activation| * sum|outgoing weights| as a bias-corrected "
            "EMA with decay_rate, which needs a forward hook to see activations.\n"
            "  2. Per-unit age, so only units past maturity_threshold are eligible.\n"
            "  3. Reinitialize the bottom replacement_rate fraction of eligible units from the "
            "init distribution, zero their OUTGOING weights, and add the unit's mean activation "
            "times those outgoing weights into the downstream bias so the layer's output is "
            "unchanged at the moment of replacement.\n"
            "  4. Zero the Adam exp_avg and exp_avg_sq rows for the replaced units, otherwise "
            "stale moments immediately undo the reinitialization.\n"
            "Note the interaction with T2: this rewrites optimizer state every step, so "
            "'carry the optimizer across cycles' means something different under CBP.")


REGISTRY = {c.kind: c for c in (NoOp, ShrinkPerturb, Dash, ContinualBackprop)}


def build(cfg_intervention):
    kind = cfg_intervention.kind
    if kind not in REGISTRY:
        raise ValueError(f"unknown intervention {kind!r}; known {sorted(REGISTRY)}")
    return REGISTRY[kind](cfg_intervention.params)


@torch.no_grad()
def _global_norm(model):
    return float(sum(p.detach().float().norm() ** 2 for p in model.parameters()) ** 0.5)
