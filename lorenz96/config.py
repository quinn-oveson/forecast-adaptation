import hashlib
import json
import platform
import socket
import subprocess
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import torch
import yaml

# Comments name the notes/DECISIONS.md entry each field decides, so a resolved config is
# readable as a filled-in decision register. Nothing consequential gets a default: a missing
# key is an error, not a silent inheritance of round 1's value.


class _Required:
    __slots__ = ()

    def __repr__(self):
        return "<required>"


REQUIRED = _Required()

# S4: chunk c of seed s is seeded s*31 + c, so seeds start sharing chunks at 32 cycles.
MAX_CYCLES = 31


@dataclass(frozen=True)
class Data:
    F: Any = REQUIRED                    # A5/E5; scalar only, per-site rejected in validate
    noise: float = REQUIRED              # O6
    history: int = REQUIRED              # D2
    n_cycles: int = REQUIRED             # S7
    chunk: int = REQUIRED                # S7
    n_val: int = REQUIRED                # S8
    cycle: int = REQUIRED                # which chunk index this run trains at
    data_mix: str = REQUIRED             # "all" | "new"   E4
    n_test: Optional[int] = None         # mechanical: None -> n_val


@dataclass(frozen=True)
class Model:
    hidden: int = REQUIRED               # N3
    n_layers: int = REQUIRED             # N2
    kernel: int = REQUIRED               # N2
    heteroscedastic: bool = False        # N4; NLL loss not implemented
    pos_embed: Optional[int] = None      # N5; not implemented, E4 prerequisite


@dataclass(frozen=True)
class Train:
    lr: float = REQUIRED                 # T6
    batch_size: int = REQUIRED           # T5
    budget_steps: int = REQUIRED         # T4
    loss_target: str = REQUIRED          # "noisy" | "clean"   O7
    weight_decay: float = REQUIRED       # T10
    grad_clip: Optional[float] = REQUIRED  # T10; null disables
    schedule: str = REQUIRED             # "constant" | "warmup"   T8
    warmup_steps: int = 0                # T8
    warmup_factor: float = 1.0           # T8; lr starts at lr/factor
    optimizer: str = "adam"              # T1
    betas: Any = (0.9, 0.999)            # T1
    drop_last: bool = True               # T5
    budget_unit: str = "steps"           # T3; only "steps" implemented
    loss: str = "mse"                    # T9


@dataclass(frozen=True)
class Eval:
    kind: str = REQUIRED                 # "log" | "linear" | "explicit"   C5
    n: Optional[int] = REQUIRED          # number of eval points; unused when explicit
    steps: Optional[Any] = None          # explicit step list
    batch_size: int = 1024               # mechanical: val forward-pass chunking


@dataclass(frozen=True)
class Diagnostics:
    enabled: bool = REQUIRED
    probe_n: Optional[int] = None            # M12
    probe_sampling: Optional[str] = None     # "first" | "random"   M12
    at: Any = ("start", "best", "final")     # mechanical: SVD cost vs resolution


@dataclass(frozen=True)
class Intervention:
    kind: str = REQUIRED                 # "none" | "snp" | "dash" | "cbp"
    params: Any = None                   # dict; each method validates its own keys


@dataclass(frozen=True)
class IO:
    out_dir: str = REQUIRED
    save: Any = REQUIRED                 # subset of best_clean / best_noisy / final   C1,C2
    init_from: Optional[str] = None       # warm-start checkpoint
    carry_optimizer: bool = False          # T2
    run_id: Optional[str] = None           # mechanical: config hash when unset
    overwrite: bool = False


@dataclass(frozen=True)
class Config:
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    train: Train = field(default_factory=Train)
    eval: Eval = field(default_factory=Eval)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    intervention: Intervention = field(default_factory=Intervention)
    io: IO = field(default_factory=IO)
    seed: int = REQUIRED
    # Left separable so S1 stays open: unset means derived from `seed` via stream.derive_seeds.
    data_seed: Optional[int] = None
    init_seed: Optional[int] = None
    shuffle_seed: Optional[int] = None
    device: str = "auto"                 # mechanical
    enforce_determinism: bool = False    # T11; recorded either way, never assumed


SAVE_TAGS = ("best_clean", "best_noisy", "final")
DIAG_PHASES = ("start", "best", "final", "every_eval")


def _build(cls, raw):
    raw = dict(raw or {})
    kwargs = {}
    for f in fields(cls):
        if is_dataclass(f.type):
            kwargs[f.name] = _build(f.type, raw.pop(f.name, {}))
        elif f.name in raw:
            kwargs[f.name] = raw.pop(f.name)
    if raw:
        raise ValueError(f"unknown config keys under {cls.__name__}: {sorted(raw)}")
    return cls(**kwargs)


def apply_override(raw, dotted, value):
    # `--set train.lr=1e-4`; parsed with yaml so lists and nulls work as written.
    keys = dotted.split(".")
    node = raw
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(value) if isinstance(value, str) else value
    return raw


def load(path, overrides=()):
    raw = yaml.safe_load(open(path)) or {}
    for dotted, value in overrides:
        apply_override(raw, dotted, value)
    cfg = _build(Config, raw)
    validate(cfg)
    return cfg


def _missing(obj, prefix=""):
    out = []
    for f in fields(obj):
        v = getattr(obj, f.name)
        name = f"{prefix}{f.name}"
        if is_dataclass(v):
            out += _missing(v, f"{name}.")
        elif isinstance(v, _Required):
            out.append(name)
    return out


def validate(cfg):
    gaps = _missing(cfg)
    if gaps:
        raise ValueError(
            "config is missing required keys (nothing consequential defaults; see "
            "notes/DECISIONS.md):\n  " + "\n  ".join(gaps))

    d, m, t, e, g, io = cfg.data, cfg.model, cfg.train, cfg.eval, cfg.diagnostics, cfg.io

    if not np.isscalar(d.F):
        raise ValueError(
            f"data.F must be a scalar, got {type(d.F).__name__}. Per-site forcing needs "
            "system.integrate to accept a per-site vector (it reads a non-scalar F as a "
            "per-step schedule and rejects the wrong length at system.py:44-46) and needs a "
            "model that is not translation-equivariant. See notes/E4_PRECOMMITMENT.md sec 7.")
    if m.pos_embed is not None:
        raise NotImplementedError(
            "model.pos_embed is a seam, not a feature: CircularCNN is weight-shared and has no "
            "positional input. Required before any site-dependent experiment; "
            "notes/E4_PRECOMMITMENT.md sec 7 bounds the width to {4, 8, 16}.")
    if m.heteroscedastic:
        raise NotImplementedError(
            "model.heteroscedastic requires an NLL loss; train.py implements MSE only (T9/N4).")

    if d.data_mix not in ("all", "new"):
        raise ValueError(f"data.data_mix must be 'all' or 'new' (E4), got {d.data_mix!r}")
    if not 0 <= d.cycle < d.n_cycles:
        raise ValueError(f"data.cycle {d.cycle} outside 0..{d.n_cycles - 1}")
    if d.n_cycles > MAX_CYCLES:
        raise ValueError(
            f"data.n_cycles={d.n_cycles} exceeds {MAX_CYCLES}: stream.py seeds chunk c of seed s "
            "as s*31+c, so at 32 cycles seed s chunk 31 collides with seed s+1 chunk 0 and "
            "different seeds silently share training data (S4).")

    if t.loss_target not in ("noisy", "clean"):
        raise ValueError(f"train.loss_target must be 'noisy' or 'clean' (O7), got {t.loss_target!r}")
    if t.optimizer != "adam":
        raise NotImplementedError(f"train.optimizer {t.optimizer!r} not implemented (T1)")
    if t.loss != "mse":
        raise NotImplementedError(f"train.loss {t.loss!r} not implemented (T9)")
    if t.budget_unit != "steps":
        raise NotImplementedError(
            f"train.budget_unit {t.budget_unit!r} not implemented; T3 stays open in the schema.")
    if t.schedule not in ("constant", "warmup"):
        raise ValueError(f"train.schedule must be 'constant' or 'warmup' (T8), got {t.schedule!r}")
    if t.schedule == "warmup" and not (t.warmup_steps > 0 and t.warmup_factor >= 1.0):
        raise ValueError("train.schedule='warmup' needs warmup_steps > 0 and warmup_factor >= 1")

    if e.kind not in ("log", "linear", "explicit"):
        raise ValueError(f"eval.kind must be log/linear/explicit (C5), got {e.kind!r}")
    if e.kind == "explicit" and not e.steps:
        raise ValueError("eval.kind='explicit' needs eval.steps")
    if e.kind != "explicit" and not e.n:
        raise ValueError(f"eval.kind={e.kind!r} needs eval.n")

    if g.enabled:
        if not g.probe_n or g.probe_sampling not in ("first", "random"):
            raise ValueError(
                "diagnostics.enabled needs probe_n and probe_sampling in {'first','random'}. "
                "M12: the inherited code used the first 256 validation windows, which overlap "
                "heavily in time and understate effective rank.")
        bad = [p for p in g.at if p not in DIAG_PHASES]
        if bad:
            raise ValueError(f"diagnostics.at has unknown phases {bad}; allowed {DIAG_PHASES}")

    bad = [s for s in io.save if s not in SAVE_TAGS]
    if bad:
        raise ValueError(f"io.save has unknown tags {bad}; allowed {SAVE_TAGS}")
    if io.carry_optimizer and not io.init_from:
        raise ValueError("io.carry_optimizer needs io.init_from (T2: nothing to carry from)")
    return cfg


def eval_step_grid(cfg):
    # Steps at which validation runs. Always includes the final step so `final` is a real eval.
    n_budget, e = cfg.train.budget_steps, cfg.eval
    if e.kind == "explicit":
        steps = [int(s) for s in e.steps]
    elif e.kind == "linear":
        steps = np.linspace(1, n_budget, int(e.n)).round().astype(int).tolist()
    else:
        steps = np.geomspace(1, n_budget, int(e.n)).round().astype(int).tolist()
    steps = sorted({s for s in steps if 1 <= s <= n_budget} | {n_budget})
    return steps


def resolved_seeds(cfg):
    # S1 stays open: each stream may be pinned independently, else all derive from `seed`.
    from .stream import derive_seeds
    derived = derive_seeds(cfg.seed, cfg.data.cycle)
    return dict(seed=cfg.seed,
                data_seed=cfg.data_seed if cfg.data_seed is not None else derived["data_seed"],
                noise_seed=derived["noise_seed"],
                init_seed=cfg.init_seed if cfg.init_seed is not None else derived["init_seed"],
                shuffle_seed=(cfg.shuffle_seed if cfg.shuffle_seed is not None
                              else derived["shuffle_seed"]))


def resolve_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _leaf(v):
    # Non-scalars become JSON in a single column so the CSV header cannot depend on a value.
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return json.dumps(v, sort_keys=True, default=str)


def flatten(cfg, prefix="cfg::"):
    out = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if is_dataclass(v):
            out.update(flatten(v, f"{prefix}{f.name}."))
        else:
            out[f"{prefix}{f.name}"] = _leaf(v)
    return out


def flat_columns(cls=Config, prefix="cfg::"):
    # Derived from the dataclass definition, so the CSV header exists before any run.
    cols = []
    for f in fields(cls):
        if is_dataclass(f.type):
            cols += flat_columns(f.type, f"{prefix}{f.name}.")
        else:
            cols.append(f"{prefix}{f.name}")
    return cols


def config_hash(cfg):
    flat = {k: v for k, v in flatten(cfg).items() if not k.startswith("cfg::io.")}
    payload = json.dumps(flat, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def provenance():
    def git(*args):
        return subprocess.run(("git",) + args, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        sha, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except Exception:
        sha, dirty = "unknown", True
    return dict(git_sha=sha, git_dirty=dirty,
                created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                host=socket.gethostname(), python_version=platform.python_version(),
                torch_version=torch.__version__, numpy_version=np.__version__)
