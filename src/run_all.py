"""Experiments: integration drift, open loop vs P/PI velocity control, cascade torque control."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from arm import Arm, euler_step, rk4_step, quintic

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
arm = Arm()
Q0 = np.deg2rad([-30.0, 60.0]); QF = np.deg2rad([60.0, 30.0]); T = 2.0

# 1. Energy drift of a free (zero torque) motion for several steps and integrators
rows = []
fig, ax = plt.subplots(figsize=(7, 4))
for name, step in [("Euler", euler_step), ("RK4", rk4_step)]:
    for Te in [0.001, 0.005, 0.010]:
        q, qd = np.array([0.5, 0.8]), np.zeros(2)
        e0 = arm.energy(q, qd); E = []
        n = int(2.0 / Te)
        for _ in range(n):
            q, qd = step(arm, q, qd, np.zeros(2), Te); E.append(arm.energy(q, qd) - e0)
        rows.append(dict(integrator=name, Te_ms=Te*1e3, max_abs_energy_drift_J=float(np.max(np.abs(E)))))
        if name == "Euler" or Te == 0.010:
            ax.plot(np.arange(1, n+1)*Te, E, label=f"{name} Te={Te*1e3:.0f} ms")
ax.set_xlabel("t (s)"); ax.set_ylabel("E(t) - E(0) (J)"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Energy drift, free motion (should be 0)")
fig.tight_layout(); fig.savefig(f"{OUT}/energy_drift.png", dpi=130); plt.close(fig)
pd.DataFrame(rows).to_csv(f"{OUT}/energy_drift.csv", index=False)

# 2. Velocity-level control (joint velocity treated as the controlled input, actuator bias b)
Te = 0.005; n = int(3.0 / Te); t = np.arange(n) * Te
bias = np.array([0.2, 0.2])
ref = np.array([quintic(Q0, QF, T, ti)[0] for ti in t]); dref = np.array([quintic(Q0, QF, T, ti)[1] for ti in t])
def vel_sim(kp, ki):
    q = Q0.copy(); I = np.zeros(2); out = []
    for k in range(n):
        e = ref[k] - q; I += e * Te
        u = dref[k] + kp * e + ki * I if (kp or ki) else dref[k]
        q = q + Te * (u + bias); out.append(q.copy())
    return np.array(out)
res = {"open loop": vel_sim(0, 0), "P (Kp=2)": vel_sim(2, 0), "PI (Kp=2, Ki=2)": vel_sim(2, 2)}
fig, ax = plt.subplots(figsize=(7, 4)); rows = []
for k, v in res.items():
    err = np.rad2deg(np.linalg.norm(v - ref, axis=1)); ax.plot(t, err, label=k)
    rows.append(dict(controller=k, final_error_deg=float(err[-1]), max_error_deg=float(err.max())))
ax.set_xlabel("t (s)"); ax.set_ylabel("|error| (deg)"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Velocity control with actuator bias 0.2 rad/s")
fig.tight_layout(); fig.savefig(f"{OUT}/velocity_control.png", dpi=130); plt.close(fig)
pd.DataFrame(rows).to_csv(f"{OUT}/velocity_control.csv", index=False)

# 3. Cascade torque control with feedforward variants, optional sensor delay
def cascade(ff, delay=0.0, Kp=4.0, Kv=np.diag([3.2, 0.8]), Ki=np.diag([6.0, 1.5]), dt=0.001, Tend=3.0):
    model = Arm(m2=1.5 * 0.8) if ff == "model m2 -20%" else arm
    nd = int(round(delay / dt)); n = int(Tend / dt)
    q, qd = Q0.copy(), np.zeros(2); I = np.zeros(2)
    hist = [(q.copy(), qd.copy())] * (nd + 1); Q = []; Er = []
    for k in range(n):
        ti = k * dt
        qr, dqr, ddqr = quintic(Q0, QF, T, ti)
        qm, qdm = hist[0] if nd else (q, qd)
        wref = Kp * (qr - qm) + dqr
        ev = wref - qdm; I += ev * dt
        tau = Kv @ ev + Ki @ I
        if ff == "G(qd)": tau += model.G(qr)
        elif ff in ("M qdd+C+G", "model m2 -20%"): tau += model.M(qr) @ ddqr + model.C(qr, dqr) + model.G(qr)
        q, qd = rk4_step(arm, q, qd, tau, dt)
        hist.append((q.copy(), qd.copy())); hist = hist[-(nd + 1):]
        Q.append(q.copy()); Er.append(qr - q)
    return np.array(Q), np.array(Er)
rows = []; fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for delay, ax in zip([0.0, 0.1], axs):
    for ff in ["none", "G(qd)", "M qdd+C+G", "model m2 -20%"]:
        Qs, Er = cascade(ff, delay)
        e = np.rad2deg(np.linalg.norm(Er, axis=1)); ok = np.isfinite(e).all() and e.max() < 90
        ax.plot(np.arange(len(e))*0.001, np.clip(e, 0, 90), label=ff)
        rows.append(dict(delay_ms=delay*1e3, feedforward=ff, max_error_deg=float(e.max()),
                         rms_error_deg=float(np.sqrt(np.mean(e**2))), final_error_deg=float(e[-1]), bounded=bool(ok)))
    ax.set_title(f"Cascade control, sensor delay {delay*1e3:.0f} ms"); ax.set_xlabel("t (s)")
    ax.set_ylabel("|error| (deg, clipped at 90)"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/cascade_control.png", dpi=130); plt.close(fig)
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/cascade_control.csv", index=False)
print(pd.DataFrame(rows).to_string()); print(pd.read_csv(f"{OUT}/energy_drift.csv")); print(pd.read_csv(f"{OUT}/velocity_control.csv"))

# 4. With a 100 ms delay the nominal gains are unstable: how far must the gains be reduced?
rows = []
for s in [1.0, 0.5, 0.25, 0.1, 0.05]:
    Qs, Er = cascade("M qdd+C+G", 0.1, Kp=4.0 * s, Kv=np.diag([3.2, 0.8]) * s, Ki=np.diag([6.0, 1.5]) * s)
    e = np.rad2deg(np.linalg.norm(Er, axis=1))
    rows.append(dict(gain_scale=s, max_error_deg=float(e.max()), final_error_deg=float(e[-1]),
                     bounded=bool(np.isfinite(e).all() and e.max() < 90)))
pd.DataFrame(rows).to_csv(f"{OUT}/delay_gain_sweep.csv", index=False)
print(pd.DataFrame(rows).to_string())
