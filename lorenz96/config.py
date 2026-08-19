import hashlib
import json
import platform
import socket
import subprocess
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional, Union, get_args, get_origin

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
    # `source` selects the provisioning path; the fields for the other path stay null so no
    # value is live but unused.
    source: str = REQUIRED               # "cycle" | "shift"
    noise: float = REQUIRED              # O6
    history: int = REQUIRED              # D2
    n_val: int = REQUIRED                # S8
    n_test: Optional[int] = None         # mechanical: None -> n_val

    # source == "cycle": equal chunks arriving at one forcing (round 1's shape)
    F: Any = None                        # A5; scalar only, per-site rejected in validate
    n_cycles: Optional[int] = None       # S7
    chunk: Optional[int] = None          # S7
    cycle: Optional[int] = None          # which chunk index this run trains at
    data_mix: Optional[str] = None       # "all" | "new"   E4, binary form

    # source == "shift": a large old-regime pool and a small new-regime one
    F_old: Any = None                    # E5 pre-shift forcing
    F_new: Any = None                    # E5 post-shift forcing
    n_old: Optional[int] = None          # S7 asymmetric arrival
    n_new: Optional[int] = None          # S7
    mix_ratio: Any = None                # E4 swept ratio: fraction of each batch drawn from
                                         # the new regime, or "natural" for pooled proportion


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


SAVE_TAGS = ("best_clean", "best_noisy", "best_next", "final")
DIAG_PHASES = ("start", "best", "final", "every_eval")


def _scalar_type(annotation):
    # Unwraps Optional[T] so a declared float still coerces when the key may be null.
    if get_origin(annotation) is Union:
        inner = [a for a in get_args(annotation) if a is not type(None)]
        return inner[0] if len(inner) == 1 else None
    return annotation


def _coerce(annotation, value):
    # YAML 1.1 reads `1e-3` as a string, not a float, so a learning rate written the obvious way
    # would reach Adam as text. Declared scalar types are enforced rather than trusted.
    if value is None or isinstance(value, _Required):
        return value
    t = _scalar_type(annotation)
    if t is bool:
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true",
                                                                                   "yes", "on")
    if t in (int, float, str):
        try:
            return t(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"expected {t.__name__}, got {value!r}") from exc
    return value


def _build(cls, raw):
    raw = dict(raw or {})
    kwargs = {}
    for f in fields(cls):
        if is_dataclass(f.type):
            kwargs[f.name] = _build(f.type, raw.pop(f.name, {}))
        elif f.name in raw:
            try:
                kwargs[f.name] = _coerce(f.type, raw.pop(f.name))
            except ValueError as exc:
                raise ValueError(f"config key {f.name!r}: {exc}") from None
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

    if d.source not in ("cycle", "shift"):
        raise ValueError(f"data.source must be 'cycle' or 'shift', got {d.source!r}")

    forcings = ("F",) if d.source == "cycle" else ("F_old", "F_new")
    for name in forcings:
        value = getattr(d, name)
        if value is None:
            raise ValueError(f"data.{name} is required when data.source={d.source!r}")
        if not np.isscalar(value):
            raise ValueError(
                f"data.{name} must be a scalar, got {type(value).__name__}. Per-site forcing "
                "needs system.integrate to accept a per-site vector (it reads a non-scalar F as "
                "a per-step schedule and rejects the wrong length at system.py:44-46) and needs "
                "a model that is not translation-equivariant. See "
                "notes/E4_PRECOMMITMENT.md sec 7.")

    if d.source == "shift":
        for name in ("n_old", "n_new", "mix_ratio"):
            if getattr(d, name) is None:
                raise ValueError(f"data.{name} is required when data.source='shift'")
        if d.mix_ratio != "natural" and not 0.0 <= float(d.mix_ratio) <= 1.0:
            raise ValueError(
                f"data.mix_ratio must be in [0, 1] or 'natural', got {d.mix_ratio!r}. It is the "
                "fraction of every batch drawn from the new regime (E4).")
        stale = [n for n in ("n_cycles", "chunk", "cycle", "data_mix") if getattr(d, n) is not None]
        if stale:
            raise ValueError(f"data.source='shift' but cycle-only keys are set: {stale}")
    if m.pos_embed is not None:
        raise NotImplementedError(
            "model.pos_embed is a seam, not a feature: CircularCNN is weight-shared and has no "
            "positional input. Required before any site-dependent experiment; "
            "notes/E4_PRECOMMITMENT.md sec 7 bounds the width to {4, 8, 16}.")
    if m.heteroscedastic:
        raise NotImplementedError(
            "model.heteroscedastic requires an NLL loss; train.py implements MSE only (T9/N4).")

    if d.source == "cycle":
        for name in ("n_cycles", "chunk", "cycle", "data_mix"):
            if getattr(d, name) is None:
                raise ValueError(f"data.{name} is required when data.source='cycle'")
        stale = [n for n in ("F_old", "F_new", "n_old", "n_new", "mix_ratio")
                 if getattr(d, n) is not None]
        if stale:
            raise ValueError(f"data.source='cycle' but shift-only keys are set: {stale}")
        _validate_cycle(d)

    _validate_rest(cfg)
    return cfg


def _validate_cycle(d):
    if d.data_mix not in ("all", "new"):
        raise ValueError(f"data.data_mix must be 'all' or 'new' (E4), got {d.data_mix!r}")
    if not 0 <= d.cycle < d.n_cycles:
        raise ValueError(f"data.cycle {d.cycle} outside 0..{d.n_cycles - 1}")
    if d.n_cycles > MAX_CYCLES:
        raise ValueError(
            f"data.n_cycles={d.n_cycles} exceeds {MAX_CYCLES}: stream.py seeds chunk c of seed s "
            "as s*31+c, so at 32 cycles seed s chunk 31 collides with seed s+1 chunk 0 and "
            "different seeds silently share training data (S4).")


def _validate_rest(cfg):
    t, e, g, io = cfg.train, cfg.eval, cfg.diagnostics, cfg.io
    if t.budget_steps < 1:
        raise ValueError(f"train.budget_steps must be >= 1, got {t.budget_steps}")
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
    # Under source='shift' there is no cycle index; 0 keeps every arm on the same init draw,
    # which is what E8 requires for a matched comparison.
    derived = derive_seeds(cfg.seed, cfg.data.cycle or 0)
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


def _plain(obj):
    # asdict leaves tuples in place (betas, diagnostics.at) and yaml.safe_dump rejects them.
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def dump(cfg, path):
    # A resolved snapshot, written once at submission. Queued array tasks read this instead of
    # cfg/*.yaml, so editing the working config while an array sits in the queue cannot silently
    # change what those tasks run.
    from dataclasses import asdict
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(_plain(asdict(cfg)), fh, sort_keys=False, default_flow_style=False)
    return path


def config_hash(cfg):
    ignored = ("cfg::io.out_dir", "cfg::io.run_id", "cfg::io.overwrite", "cfg::io.save")
    # io.init_from and io.carry_optimizer stay in: they are what separates cold_new from
    # warm_new, and excluding them would collide two different arms onto one run_id.
    flat = {k: v for k, v in flatten(cfg).items() if k not in ignored}
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
