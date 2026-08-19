import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

SCHEMA = "forecast-adaptation/checkpoint/1"

# The fields that must match for a warm start to be meaningful. in_channels is derived from
# data.history, so a history change is an architecture change and is caught here too.
ARCH_KEYS = ("hidden", "n_layers", "kernel", "heteroscedastic", "pos_embed", "in_channels")


def rng_state():
    state = dict(torch=torch.get_rng_state(), numpy=np.random.get_state())
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def arch_of(cfg):
    m = cfg.model
    return dict(hidden=m.hidden, n_layers=m.n_layers, kernel=m.kernel,
                heteroscedastic=m.heteroscedastic, pos_embed=m.pos_embed,
                in_channels=cfg.data.history)


def to_cpu(obj):
    # Snapshots are taken mid-run and may outlive the device allocation they came from.
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def snapshot(model, optimizer, step, metrics):
    # Everything needed to resume, captured at one step, detached from the live model.
    return dict(step=int(step), model=to_cpu(model.state_dict()),
                optimizer=to_cpu(optimizer.state_dict()), rng=rng_state(),
                metrics=dict(metrics))


def save(path, snap, cfg, prov, tag, seeds):
    # Fully resumable: weights, optimizer moments, RNG, and enough provenance that a checkpoint
    # found on disk can say what produced it. T2 ('carry Adam moments') is then a flag, not a
    # rewrite.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema=SCHEMA, tag=tag, step=snap["step"], model=snap["model"],
                   arch=arch_of(cfg), optimizer=snap["optimizer"], rng=snap["rng"],
                   config=asdict(cfg), provenance=dict(prov), seeds=dict(seeds),
                   metrics=snap["metrics"])
    torch.save(payload, path)
    return file_sha256(path)


def load(path):
    # weights_only=False is required: the payload deliberately carries config, provenance and
    # numpy RNG state, not just tensors.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("schema") != SCHEMA:
        raise ValueError(f"{path}: checkpoint schema {ckpt.get('schema')!r}, expected {SCHEMA!r}")
    return ckpt


def check_arch(ckpt, cfg, path):
    want, got = arch_of(cfg), ckpt.get("arch", {})
    diff = {k: (got.get(k), want[k]) for k in ARCH_KEYS if got.get(k) != want[k]}
    if diff:
        lines = "\n  ".join(f"{k}: checkpoint={a!r} config={b!r}" for k, (a, b) in diff.items())
        raise ValueError(f"cannot warm-start from {path}: architecture differs\n  {lines}")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]
