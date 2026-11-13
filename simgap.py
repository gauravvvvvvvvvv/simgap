"""simgap: measure how closely a simulator agrees with logged robot data.

The loop is the one sim-to-real work actually runs:

  1. take logged trajectories (commanded torque, measured joint angle
     and velocity) from a real robot,
  2. measure the agreement between the log and a nominal simulator,
  3. fit the simulator's physical parameters to ONE logged run,
  4. validate on runs the fit never saw, and report the agreement there,
     not on the run that was fitted.

The model is a single rotary robot joint:

    J * omega' = u(t - delay) - b * omega - c * tanh(omega / eps)
                 - quad * omega * |omega|

    J     link inertia            (kg m^2)
    b     viscous damping         (N m s / rad)
    c     Coulomb friction        (N m), smoothed with eps = 0.02 rad/s
    delay actuator latency        (s)
    quad  quadratic drag          (N m s^2 / rad^2)

`quad` is deliberately NOT one of the fitted parameters: it exists so a
caller can generate data with an effect the fitted model cannot express,
which is the normal situation on real hardware (there is always an
unmodelled effect). A fit can then agree well with the data while
hiding the unmodelled effect inside the parameters it does have, which
is why this library reports agreement and parameter values separately
and never treats a good agreement score as proof of correct physics.

Agreement score ("fit %") is the normalised fit used in system
identification: 100 * (1 - ||sim - real|| / ||real - mean(real)||).
100% is perfect, 0% means the simulator is exactly as good as guessing
the mean of the log, and negative means it is worse than that guess.

Fitting is free-run (multiple shooting over the whole trajectory is a
known next step): the simulator runs open-loop from the logged initial
state and the parameter search minimises the weighted trajectory
error. Free-run fits on long, unstable or chaotic trajectories can be
fragile; on a damped single joint over a few seconds they are well
behaved. Residuals weight omega at 0.1 relative to theta, matching the
rough 10x ratio between typical velocity and angle sensor noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ["J", "b", "c", "delay"]

# Tolerances for the "first exceeds" report, per channel.
TOLERANCES = {"theta": 0.05, "omega": 0.5}

_FRICTION_EPS = 0.02  # rad/s, Coulomb smoothing width
_OMEGA_WEIGHT = 0.1   # residual weight of omega relative to theta

_BOUNDS_LO = np.array([0.005, 0.0, 0.0, 0.0])
_BOUNDS_HI = np.array([0.5, 1.0, 0.5, 0.1])


@dataclass
class Trajectory:
    """One logged (or simulated) run: time, joint angle, joint
    velocity, and the commanded torque that produced them."""

    t: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    u: np.ndarray

    def save_csv(self, path: str) -> None:
        data = np.column_stack([self.t, self.theta, self.omega, self.u])
        np.savetxt(
            path,
            data,
            delimiter=",",
            header="t,theta,omega,u",
            comments="",
            fmt="%.8g",
        )


def simulate(params: dict, u: np.ndarray, t: np.ndarray, quad: float = 0.0) -> Trajectory:
    """Simulate the joint model with the given parameters.

    `params` holds any of J, b, c, delay (missing keys default to the
    values that switch that effect off, except J which defaults to
    0.05). `quad` is the unmodelled quadratic drag term: it is zero in
    every fitted model and non-zero only when generating stand-in
    "real" data. Integration is RK4 at the sample step of `t`, with the
    delayed input linearly interpolated on a half-step grid. The run
    starts at rest at theta = 0, as the demo's logged runs do.
    """
    J = float(params.get("J", 0.05))
    b = float(params.get("b", 0.0))
    c = float(params.get("c", 0.0))
    delay = float(params.get("delay", 0.0))

    u = np.asarray(u, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else 0.01

    # Delayed input on a half-step grid, so RK4 stages can read exact
    # stage times. Input before t = 0 is zero (actuator at rest).
    t_half = np.arange(0, n * 2 - 1) * (dt / 2.0) + t[0]
    u_half = np.interp(t_half - delay, t, u, left=0.0, right=u[-1])

    def accel(omega: float, torque: float) -> float:
        return (
            torque
            - b * omega
            - c * np.tanh(omega / _FRICTION_EPS)
            - quad * omega * abs(omega)
        ) / J

    theta = np.zeros(n)
    omega = np.zeros(n)
    th, om = 0.0, 0.0
    for i in range(n - 1):
        u0 = u_half[2 * i]
        um = u_half[2 * i + 1]
        u1 = u_half[2 * i + 2]
        k1o = accel(om, u0)
        k1t = om
        k2o = accel(om + 0.5 * dt * k1o, um)
        k2t = om + 0.5 * dt * k1o
        k3o = accel(om + 0.5 * dt * k2o, um)
        k3t = om + 0.5 * dt * k2o
        k4o = accel(om + dt * k3o, u1)
        k4t = om + dt * k3o
        th += dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        om += dt / 6.0 * (k1o + 2 * k2o + 2 * k3o + k4o)
        theta[i + 1] = th
        omega[i + 1] = om

    return Trajectory(t.copy(), theta, omega, u.copy())


def _channel_agreement(sim: np.ndarray, real: np.ndarray, t: np.ndarray, tol: float) -> dict:
    err = sim - real
    denom = np.linalg.norm(real - real.mean())
    fit_pct = 100.0 * (1.0 - np.linalg.norm(err) / denom) if denom > 0 else 0.0
    over = np.nonzero(np.abs(err) > tol)[0]
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(np.abs(err))) if len(err) else 0.0,
        "fit_pct": float(fit_pct),
        "first_exceeds_tol_s": float(t[over[0]]) if len(over) else None,
    }


def agreement(traj: Trajectory, params: dict) -> dict:
    """Agreement between one logged trajectory and the simulator run
    with `params` on the same commanded input, per channel."""
    sim = simulate(params, traj.u, traj.t)
    return {
        "theta": _channel_agreement(sim.theta, traj.theta, traj.t, TOLERANCES["theta"]),
        "omega": _channel_agreement(sim.omega, traj.omega, traj.t, TOLERANCES["omega"]),
    }


def agreement_table(runs: dict, params: dict) -> dict:
    """`agreement` for a named dict of trajectories."""
    return {name: agreement(traj, params) for name, traj in runs.items()}


@dataclass
class FitResult:
    params: dict
    ci95: dict  # half-widths of the 95% intervals, per parameter
    cost: float = 0.0
    nfev: int = 0
    message: str = field(default="")


def fit(runs: list, p0: dict) -> FitResult:
    """Fit J, b, c, delay to the given logged runs by free-run
    simulation error minimisation (scipy least_squares).

    95% intervals come from the Jacobian at the solution under the
    usual linearised, Gaussian-residual assumptions, so they describe
    the fit's precision on this data, not a guarantee about the true
    physical values, which unmodelled effects can bias, as the demo
    shows.
    """
    x0 = np.array([float(p0.get(k, 0.0)) for k in PARAM_NAMES])
    x0 = np.minimum(np.maximum(x0, _BOUNDS_LO + 1e-9), _BOUNDS_HI - 1e-9)

    def residuals(x: np.ndarray) -> np.ndarray:
        params = dict(zip(PARAM_NAMES, x))
        parts = []
        for traj in runs:
            sim = simulate(params, traj.u, traj.t)
            parts.append(sim.theta - traj.theta)
            parts.append(_OMEGA_WEIGHT * (sim.omega - traj.omega))
        return np.concatenate(parts)

    result = least_squares(
        residuals,
        x0,
        bounds=(_BOUNDS_LO, _BOUNDS_HI),
        x_scale="jac",
        method="trf",
    )

    x = result.x
    params = {k: float(v) for k, v in zip(PARAM_NAMES, x)}

    dof = max(1, len(result.fun) - len(x))
    s2 = float(np.sum(result.fun**2)) / dof
    try:
        cov = np.linalg.inv(result.jac.T @ result.jac) * s2
        half = 1.96 * np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        half = np.full(len(x), np.nan)
    ci95 = {k: float(v) for k, v in zip(PARAM_NAMES, half)}

    return FitResult(
        params=params,
        ci95=ci95,
        cost=float(result.cost),
        nfev=int(result.nfev),
        message=str(result.message),
    )


def to_json(obj, path: str) -> None:
    """Write a report dict as JSON, converting numpy scalars."""
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
