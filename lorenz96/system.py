import numpy as np

K = 40
DT = 0.01


def rhs(x, F):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def tangent_rhs(x, v):
    return (np.roll(x, 1) * (np.roll(v, -1) - np.roll(v, 2))
            + (np.roll(x, -1) - np.roll(x, 2)) * np.roll(v, 1) - v)


def rk4_step(x, F, dt=DT):
    k1 = rhs(x, F)
    k2 = rhs(x + 0.5 * dt * k1, F)
    k3 = rhs(x + 0.5 * dt * k2, F)
    k4 = rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _rk4_step_pair(x, v, F, dt=DT):
    # Advance state and tangent vector together so both see the same RK4 stages.
    k1, l1 = rhs(x, F), tangent_rhs(x, v)
    k2, l2 = rhs(x + 0.5 * dt * k1, F), tangent_rhs(x + 0.5 * dt * k1, v + 0.5 * dt * l1)
    k3, l3 = rhs(x + 0.5 * dt * k2, F), tangent_rhs(x + 0.5 * dt * k2, v + 0.5 * dt * l2)
    k4, l4 = rhs(x + dt * k3, F), tangent_rhs(x + dt * k3, v + dt * l3)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    v_next = v + (dt / 6.0) * (l1 + 2 * l2 + 2 * l3 + l4)
    return x_next, v_next


def initial_state(F=8.0, seed=0, k=K):
    # Small perturbation off the unstable fixed point x_k = F.
    rng = np.random.default_rng(seed)
    return np.full(k, float(F)) + 0.01 * rng.standard_normal(k)


def integrate(x0, F, n_steps, dt=DT, spinup=0):
    # F may be a scalar or a per-step array of length spinup + n_steps.
    total = spinup + n_steps
    F_seq = np.full(total, float(F)) if np.isscalar(F) else np.asarray(F, dtype=float)
    if F_seq.shape[0] != total:
        raise ValueError(f"F schedule has length {F_seq.shape[0]}, expected {total}")

    x = np.asarray(x0, dtype=float).copy()
    for t in range(spinup):
        x = rk4_step(x, F_seq[t], dt)

    traj = np.empty((n_steps, x.shape[0]))
    for t in range(n_steps):
        traj[t] = x
        x = rk4_step(x, F_seq[spinup + t], dt)
    return traj


def lyapunov_exponent(F=8.0, n_steps=100_000, dt=DT, spinup=5_000, seed=0):
    # Benettin: evolve a tangent vector, renormalize each step, average log growth.
    x = initial_state(F, seed)
    for _ in range(spinup):
        x = rk4_step(x, F, dt)

    rng = np.random.default_rng(seed + 1)
    v = rng.standard_normal(x.shape[0])
    v /= np.linalg.norm(v)

    log_growth = 0.0
    for _ in range(n_steps):
        x, v = _rk4_step_pair(x, v, F, dt)
        norm = np.linalg.norm(v)
        log_growth += np.log(norm)
        v /= norm
    return log_growth / (n_steps * dt)


def f_schedule(kind, n_steps, F0=8.0, F1=16.0, width=0, n_cycles=2):
    # width=0 reproduces a step; larger width linearly ramps F0 to F1 about the midpoint.
    if kind == "constant":
        return np.full(n_steps, float(F0))
    if kind == "cyclic":
        # Revisits both regimes so forgetting and relearning can be measured separately.
        period = n_steps / n_cycles
        square = np.sign(np.sin(2 * np.pi * np.arange(n_steps) / period))
        if width > 0:
            k = np.ones(int(width)) / int(width)
            square = np.convolve(square, k, mode="same")
        return F0 + (F1 - F0) * (square + 1) / 2
    if kind not in ("step", "ramp"):
        raise ValueError(f"unknown schedule kind: {kind}")

    width = 0 if kind == "step" else int(width)
    F_seq = np.full(n_steps, float(F0))
    mid = n_steps // 2
    start, end = mid - width // 2, mid + (width - width // 2)
    F_seq[end:] = float(F1)
    if width > 0:
        F_seq[start:end] = np.linspace(F0, F1, end - start, endpoint=False)
    return F_seq
