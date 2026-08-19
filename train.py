#!/usr/bin/env python
"""Train a CircularCNN to forecast Lorenz 96, for one (cycle, arm, seed, lr) cell.

One invocation trains one model against one data mix and emits one row. It computes losses,
diagnostics and checkpoints -- never forecast metrics. Rollout, NRMSE and valid prediction time
belong to a separate evaluator that loads the checkpoints written here, so that every entry in
the M section of notes/DECISIONS.md (and the M4 denominator bug) stays out of the training path.

Cycles, arms and sweeps are the caller's job: `train_once` takes the data it should train on,
and a driver decides which checkpoint gets handed to the next cycle.
"""
import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from lorenz96 import checkpoint as ckpt_mod
from lorenz96 import config as cfgmod
from lorenz96 import diagnostics, interventions, rows
from lorenz96.models import CircularCNN, count_params
from lorenz96.shift import MixSampler, ShiftStream, resolve_ratio
from lorenz96.stream import CycleStream, Split

# Shrink & Perturb draws its perturbation from a fresh init; offset so it is not the same draw
# as the model's own initialization while staying a deterministic function of the seed.
PERTURB_SEED_OFFSET = 7_919

# The diagnostic probe draws from its own generator: sharing the batch sampler's would make
# turning diagnostics on change the training batch order, so a measurement would perturb the
# thing it measures.
PROBE_SEED_OFFSET = 104_729


@dataclass
class Result:
    run: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    diag: list = field(default_factory=list)
    snapshots: dict = field(default_factory=dict)   # save tag -> checkpoint snapshot


def build_model(cfg, device, init_seed):
    # in_channels is the history depth: input is (n, history, K) and Conv1d reads history as
    # channels. A history change is therefore an architecture change.
    torch.manual_seed(init_seed)
    model = CircularCNN(hidden=cfg.model.hidden, n_layers=cfg.model.n_layers,
                        kernel=cfg.model.kernel, in_channels=cfg.data.history,
                        heteroscedastic=cfg.model.heteroscedastic)
    return model.to(device)


def lr_at(cfg, step):
    # T8: the label must describe what ran, so the realized value is logged at every eval point.
    t = cfg.train
    if t.schedule == "constant":
        return t.lr
    frac = min(1.0, step / t.warmup_steps)
    return t.lr * t.warmup_factor ** (frac - 1.0)


def batch_indices(n, batch_size, drop_last, generator):
    # Reshuffled every epoch from shuffle_seed; steps are counted globally, not per epoch (T3).
    limit = (n // batch_size) * batch_size if drop_last else n
    if limit == 0:
        raise ValueError(f"{n} training windows < batch_size {batch_size} with drop_last=True")
    while True:
        perm = torch.randperm(n, generator=generator)
        for i in range(0, limit, batch_size):
            yield perm[i:i + batch_size]


@torch.no_grad()
def val_losses(model, split, batch_size):
    # Three references, all logged, so no criterion is privileged without leaving a record:
    #   clean  MSE(pred, clean tendency)   what the inherited pipeline measured
    #   noisy  MSE(pred, noisy tendency)   the deployable tendency criterion
    #   next   MSE(x[:,-1] + pred, x_clean[:,-1] + y_clean)   <- selection uses this
    # `next` is what the rollout actually needs, and unlike `clean` it does not penalise the
    # model for denoising its own input: given a noisy x, the optimal tendency is
    # E[c_t+1 | x] - x, which contains -eps_t by construction. It is also invariant to the
    # tendency-vs-next-state parameterisation, so it survives a change to D1.
    model.eval()
    tot_clean = tot_noisy = tot_next = 0.0
    n = split.x.shape[0]
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        pred = model(split.x[sl])
        k = pred.shape[0]
        tot_clean += float(((pred - split.y_clean[sl]) ** 2).mean()) * k
        tot_noisy += float(((pred - split.y[sl]) ** 2).mean()) * k
        err = (split.x[sl][:, -1] + pred) - (split.x_clean[sl][:, -1] + split.y_clean[sl])
        tot_next += float((err ** 2).mean()) * k
    model.train()
    return tot_clean / n, tot_noisy / n, tot_next / n


def probe_batch(cfg, split, seed):
    # M12: the inherited code used the first 256 validation windows, which overlap in time and
    # understate effective rank. Both the count and the sampling are config, never assumed.
    n_avail = split.x.shape[0]
    n = min(int(cfg.diagnostics.probe_n), n_avail)
    if cfg.diagnostics.probe_sampling == "first":
        idx = torch.arange(n)
    else:
        gen = torch.Generator().manual_seed(seed + PROBE_SEED_OFFSET)
        idx = torch.randperm(n_avail, generator=gen)[:n]
    return split.x[idx.to(split.x.device)]


def diag_rows(run_id, step, phase, values):
    return [dict(run_id=run_id, step=step, phase=phase, metric=k, value=v)
            for k, v in values.items()]


def grad_over(model, split, cfg):
    # One full-data gradient pass, for interventions that need a descent direction before the
    # first step (DASH). Accumulated as a mean over batches so it matches the training loss.
    target = split.y if cfg.train.loss_target == "noisy" else split.y_clean
    n, bs = split.x.shape[0], cfg.train.batch_size
    model.zero_grad(set_to_none=True)
    for i in range(0, n, bs):
        pred = model(split.x[i:i + bs])
        loss = ((pred - target[i:i + bs]) ** 2).mean() * (pred.shape[0] / n)
        loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()
             if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return grads


def train_once(cfg, train_split, val_split, run_id=None, init_ckpt=None, init_from=None,
               device=None, prov=None, val_old=None, sampler=None):
    # val_split is the selection set. val_old, when given, is the pre-shift regime: logged at
    # every eval point but never selected on, so forgetting is measured without steering the
    # checkpoint choice. sampler overrides uniform batching (see shift.MixSampler).
    device = device or cfgmod.resolve_device(cfg.device)
    # A driver looping thousands of cells should resolve provenance once and pass it in; a
    # one-off library call still gets a complete row rather than blank columns.
    prov = cfgmod.provenance() if prov is None else prov
    seeds = cfgmod.resolved_seeds(cfg)
    run_id = run_id or cfgmod.config_hash(cfg)
    if cfg.enforce_determinism:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

    train_split = Split(*(t.to(device) for t in train_split))
    val_split = Split(*(t.to(device) for t in val_split))
    if val_old is not None:
        val_old = Split(*(t.to(device) for t in val_old))

    model = build_model(cfg, device, seeds["init_seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                                 betas=tuple(cfg.train.betas),
                                 weight_decay=cfg.train.weight_decay)

    warm = dict(init_from=init_from or "", init_ckpt_sha="", init_ckpt_step="")
    if init_ckpt is not None:
        ckpt_mod.check_arch(init_ckpt, cfg, init_from)
        model.load_state_dict({k: v.to(device) for k, v in init_ckpt["model"].items()},
                              strict=True)
        if cfg.io.carry_optimizer:
            optimizer.load_state_dict(init_ckpt["optimizer"])
        warm["init_ckpt_step"] = init_ckpt.get("step", "")
        warm["init_ckpt_sha"] = ckpt_mod.file_sha256(init_from) if init_from else ""

    # Weights are loaded first, then the optimizer, then the intervention: an intervention that
    # rewrites optimizer state (CBP) must see the state it is meant to rewrite.
    intervention = interventions.build(cfg.intervention)
    gen = torch.Generator().manual_seed(seeds["shuffle_seed"])
    fresh_state = None
    if cfg.intervention.kind != "none":
        fresh_state = build_model(cfg, "cpu",
                                  seeds["init_seed"] + PERTURB_SEED_OFFSET).state_dict()
    grad_ema = grad_over(model, train_split, cfg) if intervention.requires_grad_pass() else None
    intervention.on_run_start(interventions.RunStart(
        model=model, optimizer=optimizer, fresh_state=fresh_state, generator=gen,
        grad_ema=grad_ema))

    # Captured after warm start and intervention, so wratio:: means drift within this run for
    # every arm alike. M14's cold/warm asymmetry cannot arise; the checkpoint stores absolute
    # norms, so cross-cycle ratios remain computable downstream.
    init_norms = diagnostics.layer_norms(model)

    diag, trace, snapshots = [], [], {}
    diag += diag_rows(run_id, 0, "intervention", intervention.log())
    probe = (probe_batch(cfg, val_split, seeds["shuffle_seed"])
             if cfg.diagnostics.enabled else None)
    if cfg.diagnostics.enabled and "start" in cfg.diagnostics.at:
        diag += diag_rows(run_id, 0, "start", diagnostics.collect(model, probe, init_norms))

    eval_at = set(cfgmod.eval_step_grid(cfg))
    last_eval = max(eval_at)
    target = train_split.y if cfg.train.loss_target == "noisy" else train_split.y_clean
    batches = (iter(sampler) if sampler is not None else
               batch_indices(train_split.x.shape[0], cfg.train.batch_size, cfg.train.drop_last,
                             gen))
    best = {k: (float("inf"), 0, 0.0) for k in ("clean", "noisy", "next")}
    best_old = {}
    ema = None
    model.train()
    t0 = time.perf_counter()

    for step in range(1, cfg.train.budget_steps + 1):
        lr = lr_at(cfg, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        idx = next(batches)
        pred = model(train_split.x[idx])
        loss = ((pred - target[idx]) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        intervention.on_step_end(model, optimizer, step)

        train_loss = float(loss.detach())
        ema = train_loss if ema is None else 0.99 * ema + 0.01 * train_loss

        if step in eval_at:
            vc, vn, vx = val_losses(model, val_split, cfg.eval.batch_size)
            old = (None if val_old is None else
                   val_losses(model, val_old, cfg.eval.batch_size))
            trace.append(dict(run_id=run_id, step=step, lr_realized=lr, train_loss_ema=ema,
                              val_clean=vc, val_noisy=vn, val_next=vx,
                              val_old_clean="" if old is None else old[0],
                              val_old_noisy="" if old is None else old[1],
                              val_next_old="" if old is None else old[2]))
            for key, value in (("clean", vc), ("noisy", vn), ("next", vx)):
                if value < best[key][0]:
                    best[key] = (value, step, lr)
                    # Recorded at the SELECTED step, so the old-regime cost reported alongside
                    # a checkpoint is the cost of that checkpoint.
                    if key == "next" and old is not None:
                        best_old = dict(val_old_clean_at_best=old[0],
                                        val_old_noisy_at_best=old[1],
                                        val_next_old_at_best=old[2])
                    tag = f"best_{key}"
                    if tag in cfg.io.save:
                        snapshots[tag] = ckpt_mod.snapshot(
                            model, optimizer, step,
                            dict(val_clean=vc, val_noisy=vn, val_next=vx, train_loss_ema=ema,
                                 lr_realized=lr))
            if cfg.diagnostics.enabled and "every_eval" in cfg.diagnostics.at:
                diag += diag_rows(run_id, step, "eval",
                                  diagnostics.collect(model, probe, init_norms))

    wall = time.perf_counter() - t0
    final_clean, final_noisy, final_next = val_losses(model, val_split, cfg.eval.batch_size)
    final_old = ({} if val_old is None else
                 dict(zip(("final_val_old_clean", "final_val_old_noisy", "final_val_next_old"),
                          val_losses(model, val_old, cfg.eval.batch_size))))
    if "final" in cfg.io.save:
        snapshots["final"] = ckpt_mod.snapshot(
            model, optimizer, cfg.train.budget_steps,
            dict(val_clean=final_clean, val_noisy=final_noisy, val_next=final_next,
                 train_loss_ema=ema, lr_realized=lr_at(cfg, cfg.train.budget_steps)))

    final_diag = {}
    if cfg.diagnostics.enabled:
        final_diag = diagnostics.collect(model, probe, init_norms)
        if "final" in cfg.diagnostics.at:
            diag += diag_rows(run_id, cfg.train.budget_steps, "final", final_diag)
        if "best" in cfg.diagnostics.at:
            for tag, snap in snapshots.items():
                if not tag.startswith("best_"):
                    continue
                probe_model = build_model(cfg, device, seeds["init_seed"])
                probe_model.load_state_dict({k: v.to(device) for k, v in snap["model"].items()})
                diag += diag_rows(run_id, snap["step"], tag,
                                  diagnostics.collect(probe_model, probe, init_norms))

    # The windows the sampler can actually reach, not the size of the pooled tensor: at
    # ratio 1.0 the old half is present in memory but unreachable, and epochs_equiv would lie.
    n_train = train_split.x.shape[0]
    if sampler is not None:
        n_train = ((sampler.n_old if sampler.n_old_per else 0)
                   + (sampler.n_new if sampler.n_new_per else 0))
    run = dict(run_id=run_id, schema_version=rows.SCHEMA_VERSION,
               config_hash=cfgmod.config_hash(cfg), device=device,
               determinism_enforced=cfg.enforce_determinism,
               n_train=n_train, n_val=val_split.x.shape[0], n_params=count_params(model),
               chunks_used=cfg.data.cycle + 1 if cfg.data.data_mix == "all" else 1,
               carry_optimizer=cfg.io.carry_optimizer,
               intervention_kind=cfg.intervention.kind,
               intervention_params=json.dumps(cfg.intervention.params or {}, sort_keys=True),
               status="ok", error="",
               wall_seconds=round(wall, 3), steps_run=cfg.train.budget_steps,
               samples_seen=cfg.train.budget_steps * cfg.train.batch_size,
               epochs_equiv=round(cfg.train.budget_steps * cfg.train.batch_size / n_train, 4),
               train_loss_final=train_loss, train_loss_ema_final=ema,
               final_val_clean=final_clean, final_val_noisy=final_noisy,
               final_val_next=final_next,
               diag_effective_rank_final=final_diag.get("effective_rank", ""),
               diag_dormant_frac_final=final_diag.get("dormant_frac", ""),
               diag_weight_norm_final=final_diag.get("weight_norm", ""),
               saved_tags=";".join(sorted(snapshots)), ckpt_dir="")
    run.update(prov)
    run.update(seeds)
    run.update(warm)
    run.update(final_old)
    run.update(best_old)
    if sampler is not None:
        run.update(sampler.log())
    run.update(cfgmod.flatten(cfg))
    for key in ("clean", "noisy", "next"):
        value, step, lr = best[key]
        # C6: an optimum at the last eval point means the budget, not the data, chose the model.
        run[f"best_val_{key}"] = value
        run[f"step_of_best_{key}"] = step
        run[f"lr_at_best_{key}"] = lr
        run[f"best_at_boundary_{key}"] = step == last_eval
    return Result(run=run, trace=trace, diag=diag, snapshots=snapshots)


def build_stream(cfg, seeds):
    d = cfg.data
    if d.source == "shift":
        return ShiftStream(seed=seeds["data_seed"], noise=d.noise, history=d.history,
                           F_old=d.F_old, F_new=d.F_new, n_old=d.n_old, n_new=d.n_new,
                           n_val=d.n_val)
    return CycleStream(seed=seeds["data_seed"], noise=d.noise, n_cycles=d.n_cycles,
                       chunk=d.chunk, n_val=d.n_val, history=d.history, F=d.F,
                       n_test=d.n_test)


def splits_for(cfg, stream, seeds):
    # Returns (train, val, val_old, sampler). Under 'shift' the training split is the pooled
    # old+new tensor and the sampler decides what fraction of each batch comes from where.
    if cfg.data.source != "shift":
        return stream.cycle_data(cfg.data.cycle, data_all=cfg.data.data_mix == "all"), \
            stream.val, None, None
    ratio = resolve_ratio(cfg.data.mix_ratio, stream.n_old, stream.n_new)
    sampler = MixSampler(stream.n_old, stream.n_new, ratio, cfg.train.batch_size,
                         torch.Generator().manual_seed(seeds["shuffle_seed"]))
    return stream.pooled(), stream.val_new, stream.val_old, sampler


ALIASES = [("--seed", "seed"), ("--cycle", "data.cycle"), ("--mix", "data.data_mix"),
           ("--mix-ratio", "data.mix_ratio"),
           ("--lr", "train.lr"), ("--noise", "data.noise"), ("--out-dir", "io.out_dir"),
           ("--init-from", "io.init_from"), ("--run-id", "io.run_id"), ("--device", "device")]


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="YAML config; the canonical record of a run")
    ap.add_argument("--set", action="append", default=[], metavar="DOTTED=VALUE",
                    help="override any config key, e.g. --set train.grad_clip=1.0")
    for flag, _ in ALIASES:
        ap.add_argument(flag)
    ap.add_argument("--carry-optimizer", action="store_true",
                    help="warm-start Adam moments too, not just weights (T2)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and validate the config, build nothing")
    return ap.parse_args(argv)


def overrides_from(args):
    out = []
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects DOTTED=VALUE, got {item!r}")
        out.append(tuple(item.split("=", 1)))
    for flag, dotted in ALIASES:
        value = getattr(args, flag.lstrip("-").replace("-", "_"))
        if value is not None:
            out.append((dotted, value))
    if args.carry_optimizer:
        out.append(("io.carry_optimizer", "true"))
    if args.overwrite:
        out.append(("io.overwrite", "true"))
    return out


def main(argv=None):
    args = parse_args(argv)
    cfg = cfgmod.load(args.config, overrides_from(args))
    seeds = cfgmod.resolved_seeds(cfg)
    run_id = cfg.io.run_id or cfgmod.config_hash(cfg)
    out_dir = Path(cfg.io.out_dir)
    ckpt_dir = out_dir / "checkpoints" / run_id

    if args.dry_run:
        grid = cfgmod.eval_step_grid(cfg)
        print(json.dumps(dict(run_id=run_id, config_hash=cfgmod.config_hash(cfg),
                              device=cfgmod.resolve_device(cfg.device), seeds=seeds,
                              eval_points=len(grid), eval_first=grid[:5], eval_last=grid[-3:],
                              checkpoints=[f"{ckpt_dir}/{t}.pt" for t in sorted(cfg.io.save)],
                              runs_csv=str(out_dir / "runs.csv")), indent=2, default=str))
        return 0

    if ckpt_dir.exists() and any(ckpt_dir.iterdir()) and not cfg.io.overwrite:
        raise SystemExit(f"{ckpt_dir} already holds checkpoints for run_id {run_id}; the config "
                         "hash says this is the same cell. Pass --overwrite to replace it.")

    prov = cfgmod.provenance()
    stream = build_stream(cfg, seeds)
    train_split, val_split, val_old, sampler = splits_for(cfg, stream, seeds)
    init_ckpt = ckpt_mod.load(cfg.io.init_from) if cfg.io.init_from else None

    try:
        result = train_once(cfg, train_split, val_split, run_id=run_id, init_ckpt=init_ckpt,
                            init_from=cfg.io.init_from, prov=prov, val_old=val_old,
                            sampler=sampler)
    except Exception as exc:                                    # noqa: BLE001
        failed = {c: "" for c in rows.RUN_COLUMNS}
        failed.update(cfgmod.flatten(cfg))
        failed.update(seeds)
        failed.update(prov)
        failed.update(run_id=run_id, schema_version=rows.SCHEMA_VERSION,
                      config_hash=cfgmod.config_hash(cfg), status="error",
                      error=f"{type(exc).__name__}: {exc}"[:500])
        rows.append(out_dir / "runs.csv", rows.RUN_COLUMNS, [failed])
        raise

    for tag, snap in result.snapshots.items():
        ckpt_mod.save(ckpt_dir / f"{tag}.pt", snap, cfg, prov, tag, seeds)
    result.run["ckpt_dir"] = str(ckpt_dir)

    rows.append(out_dir / "runs.csv", rows.RUN_COLUMNS, [result.run])
    rows.append(out_dir / "trace.csv", rows.TRACE_COLUMNS, result.trace)
    if result.diag:
        rows.append(out_dir / "diag.csv", rows.DIAG_COLUMNS, result.diag)

    r = result.run
    print(f"{run_id}  {r['device']}  {r['wall_seconds']}s  "
          f"best_clean={r['best_val_clean']:.6f}@{r['step_of_best_clean']}  "
          f"best_noisy={r['best_val_noisy']:.6f}@{r['step_of_best_noisy']}  "
          f"final_clean={r['final_val_clean']:.6f}  saved={r['saved_tags'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
