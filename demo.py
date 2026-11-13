"""Demo on SYNTHETIC data (no real robot here).

'Real' robot = ground-truth physics the simulator does not know about:
  quadratic drag (NOT in the simulator's model, so a gap must remain), heavier link, more viscous damping, Coulomb friction, 30 ms actuator latency, noisy sensors.
'Sim' = the nominal model an engineer would write from a CAD file / datasheet:
  lighter link, little damping, no Coulomb friction, no latency.

Steps: measure the gap -> fit sim to ONE real run -> validate on runs the fit never saw.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import simgap as sg

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(7)
DT, T = 0.01, 8.0
t = np.arange(0, T, DT)

TRUE = {"J": 0.050, "b": 0.12, "c": 0.08, "delay": 0.030}      # the real robot
UNMODELLED_QUAD = 0.04   # quadratic drag the simulator has no term for
NOMINAL = {"J": 0.040, "b": 0.05, "c": 0.00, "delay": 0.000}    # the simulator we start with


def excitation_multisine(t):
    return 0.22 * np.sin(2 * np.pi * 0.5 * t) + 0.14 * np.sin(2 * np.pi * 1.3 * t + 1.0) + 0.09 * np.sin(2 * np.pi * 2.7 * t + 2.0)


def excitation_chirp(t):
    f0, f1 = 0.2, 3.0
    return 0.30 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * t[-1])))


def excitation_pulses(t):
    u = np.zeros_like(t)
    for start, amp in [(0.5, 0.28), (2.0, -0.32), (3.6, 0.22), (5.0, -0.25), (6.4, 0.30)]:
        u[(t >= start) & (t < start + 0.9)] = amp
    return u


def measure_real(u):
    """Run the true robot and add sensor noise (what a logger would record)."""
    tr = sg.simulate(TRUE, u, t, quad=UNMODELLED_QUAD)
    return sg.Trajectory(t, tr.theta + rng.normal(0, 0.002, len(t)),
                         tr.omega + rng.normal(0, 0.02, len(t)), u)


runs = {
    "train: multisine": measure_real(excitation_multisine(t)),
    "held-out: chirp": measure_real(excitation_chirp(t)),
    "held-out: pulses": measure_real(excitation_pulses(t)),
}
for name, r in runs.items():
    r.save_csv(os.path.join(OUT, name.replace(": ", "_").replace(" ", "_") + "_real.csv"))

# 1) the gap, before any fitting
before = sg.agreement_table(runs, NOMINAL)

# 2) fit to the training run ONLY
fit = sg.fit([runs["train: multisine"]], p0=NOMINAL)

# 3) agreement after fitting, including runs the fit never saw
after = sg.agreement_table(runs, fit.params)

report = {"true_params_synthetic_only": TRUE, "nominal_params": NOMINAL,
          "fitted_params": fit.params, "fitted_ci95": fit.ci95,
          "before": before, "after": after}
sg.to_json(report, os.path.join(OUT, "report.json"))

# ---------------------------------------------------------------- figure
C_REAL, C_NOM, C_FIT = "#222222", "#d1495b", "#2e86ab"
fig = plt.figure(figsize=(12, 8.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.8, 1.0], hspace=0.55, wspace=0.28)

held = runs["held-out: chirp"]
sim_nom = sg.simulate(NOMINAL, held.u, held.t)
sim_fit = sg.simulate(fit.params, held.u, held.t)

ax = fig.add_subplot(gs[0, :])
ax.plot(held.t, held.theta, color=C_REAL, lw=1.6, label="real (logged)")
ax.plot(held.t, sim_nom.theta, color=C_NOM, lw=1.3, ls="--", label="sim: nominal")
ax.plot(held.t, sim_fit.theta, color=C_FIT, lw=1.3, label="sim: fitted to a DIFFERENT run")
ax.set_title("Held-out chirp: joint angle (the fit never saw this trajectory)", loc="left", fontsize=11)
ax.set_ylabel("theta (rad)"); ax.legend(loc="upper right", fontsize=8, ncol=3); ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[1, :])
ax.plot(held.t, np.abs(sim_nom.theta - held.theta), color=C_NOM, lw=1.2, label="nominal |error|")
ax.plot(held.t, np.abs(sim_fit.theta - held.theta), color=C_FIT, lw=1.2, label="fitted |error|")
ax.axhline(0.05, color="#888", ls=":", lw=1); ax.text(0.05, 0.053, "tolerance 0.05 rad", fontsize=7, color="#555")
ax.set_ylabel("|error| (rad)"); ax.set_xlabel("time (s)"); ax.legend(fontsize=8); ax.grid(alpha=0.25)

ax = fig.add_subplot(gs[2, 0])
names = list(runs.keys()); x = np.arange(len(names)); w = 0.38
fb = [before[n]["theta"]["fit_pct"] for n in names]
fa = [after[n]["theta"]["fit_pct"] for n in names]
ax.bar(x - w / 2, fb, w, color=C_NOM, label="nominal"); ax.bar(x + w / 2, fa, w, color=C_FIT, label="fitted")
for xi, v in zip(x - w / 2, fb): ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
for xi, v in zip(x + w / 2, fa): ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([n.replace(": ", "\n") for n in names], fontsize=8)
ax.set_ylim(min(0, min(fb) - 5), 108); ax.set_ylabel("agreement, fit % (theta)")
ax.set_title("Agreement before / after", loc="left", fontsize=11); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

ax = fig.add_subplot(gs[2, 1]); ax.axis("off")
rows = [["param", "nominal", "fitted ±95%", "true*"]]
for k in sg.PARAM_NAMES:
    rows.append([k, f"{NOMINAL[k]:.3f}", f"{fit.params[k]:.3f} ± {fit.ci95[k]:.3f}", f"{TRUE[k]:.3f}"])
tb = ax.table(cellText=rows, loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
for j in range(4): tb[0, j].set_facecolor("#e6eef2")
ax.set_title("Identified parameters", loc="left", fontsize=11)
ax.text(0, -0.02, "*true values exist only because this demo is synthetic", fontsize=7, color="#555", transform=ax.transAxes)

fig.suptitle("simgap: sim-to-real agreement, fitted on one run and validated on others (synthetic demo)", fontsize=12, y=0.98)
fig.savefig(os.path.join(OUT, "agreement_report.png"), dpi=140, bbox_inches="tight")

# ---------------------------------------------------------------- console summary
print(f"{'run':<20}{'nominal fit%':>14}{'fitted fit%':>14}{'first >0.05rad: nominal -> fitted':>40}")
for n in names:
    a, b_ = before[n]["theta"], after[n]["theta"]
    fmt = lambda v: "never" if v is None else f"{v:.2f}s"
    print(f"{n:<20}{a['fit_pct']:>13.1f}%{b_['fit_pct']:>13.1f}%{fmt(a['first_exceeds_tol_s']):>22} -> {fmt(b_['first_exceeds_tol_s'])}")
print("\nparams (fitted ± 95% | true):")
for k in sg.PARAM_NAMES:
    print(f"  {k:<6}{fit.params[k]:.4f} ± {fit.ci95[k]:.4f}   | true {TRUE[k]:.4f}")
