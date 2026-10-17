# 2-link arm: dynamics, numerical integration and cascade control

Python simulation of a planar 2-link arm (shoulder + elbow, vertical plane) built from the
exercises of a robotics tutorial (TD "Commande Robot", Master Robotics RAM, UVSQ).
The goal is to understand *why* each ingredient of a robot controller is needed by switching it on and off.

The tutorial is a course exercise; this repository is my own Python implementation of it (the original is not reproduced here).

## Model
`M(q) q̈ + C(q, q̇) + G(q) = Γ` with L = 0.3 m, m1 = 2 kg, m2 = 1.5 kg, I = mL²/12, centres of mass at L/2, g = 9.81 m/s².
Standard 2R equations (see `src/arm.py`). Reference: quintic (minimum-jerk) trajectory (-30°, 60°) → (60°, 30°) in 2 s.

## Experiments (`python src/run_all.py`, results in `results/`)

**1. Integration step.** Free motion, no torque: energy must stay constant.

| integrator | Te | max energy drift |
|---|---|---|
| explicit Euler | 1 ms / 5 ms / 10 ms | 0.82 / 4.8 / 9.4 J |
| RK4 | 1 ms / 5 ms / 10 ms | 1e-6 / 8e-4 / 2e-2 J |

Explicit Euler injects energy and the drift grows with Te; RK4 is orders of magnitude better at the same step.

**2. Velocity-level control with an actuator bias of 0.2 rad/s.** Final tracking error: open loop 48.6°, P (Kp = 2) 8.1°, PI (Kp = Ki = 2) 0.1°.
A P controller leaves a steady-state error bias/Kp per joint (0.1 rad = 5.7°, i.e. 8.1° as a norm over 2 joints); the integral term removes it.

**3. Cascade torque control** (outer position P loop Kp = 4, inner velocity PI Kv = diag(3.2, 0.8), Ki = diag(6, 1.5)), no delay:

| feedforward | max error | final error |
|---|---|---|
| none | 38.9° | 24.3° |
| G(q_d) | 10.5° | 8.5° |
| M q̈_d + C + G (exact model) | 0.09° | 0.006° |
| same, model with m2 20 % too low | 8.4° | 5.9° |

The model-based feedforward does most of the work; feedback only has to correct model error.

**4. Sensor delay of 100 ms: negative result.** With these gains the loop is unstable in my simulation, and I found no gain
setting in a coarse sweep (`results/delay_gain_sweep.csv`, plus a Kp × Kv × Ki grid I ran separately) that keeps the error bounded:
lower gains cannot hold the arm against gravity, higher gains are destabilised by the delay. This is a limitation of this simple
structure; a prediction-based compensator (e.g. Smith predictor) would be the next thing to try. I have not implemented it.

## Limitations
- Simulation only; no friction, no actuator saturation or dynamics.
- Parameters and gains are those I used from the exercise sheet as I understood it; the model equations are the standard ones, not copied from the sheet.
- Code written with the help of an AI assistant (Claude); I reviewed it and ran every experiment.

## Run
```
pip install -r requirements.txt
python src/run_all.py
```
