"""Planar 2-link arm in a vertical plane: dynamics M(q)qdd + C(q,qd) + G(q) = tau."""
import numpy as np

class Arm:
    def __init__(self, L=0.3, m1=2.0, m2=1.5, g=9.81):
        self.l1 = self.l2 = L
        self.m1, self.m2, self.g = m1, m2, g
        self.lc1 = self.lc2 = L / 2
        self.I1 = m1 * L**2 / 12
        self.I2 = m2 * L**2 / 12

    def M(self, q):
        c2 = np.cos(q[1])
        m12 = self.I2 + self.m2 * (self.lc2**2 + self.l1 * self.lc2 * c2)
        m11 = (self.I1 + self.I2 + self.m1 * self.lc1**2
               + self.m2 * (self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * c2))
        m22 = self.I2 + self.m2 * self.lc2**2
        return np.array([[m11, m12], [m12, m22]])

    def C(self, q, qd):
        h = self.m2 * self.l1 * self.lc2 * np.sin(q[1])
        return np.array([-h * (2 * qd[0] * qd[1] + qd[1]**2), h * qd[0]**2])

    def G(self, q):
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.cos(q[0]) \
             + self.m2 * self.lc2 * self.g * np.cos(q[0] + q[1])
        g2 = self.m2 * self.lc2 * self.g * np.cos(q[0] + q[1])
        return np.array([g1, g2])

    def accel(self, q, qd, tau):
        return np.linalg.solve(self.M(q), tau - self.C(q, qd) - self.G(q))

    def energy(self, q, qd):
        kin = 0.5 * qd @ self.M(q) @ qd
        y1 = self.lc1 * np.sin(q[0])
        y2 = self.l1 * np.sin(q[0]) + self.lc2 * np.sin(q[0] + q[1])
        return kin + self.g * (self.m1 * y1 + self.m2 * y2)

def euler_step(arm, q, qd, tau, dt):
    qdd = arm.accel(q, qd, tau)
    return q + dt * qd, qd + dt * qdd

def rk4_step(arm, q, qd, tau, dt):
    def f(x):
        return np.concatenate([x[2:], arm.accel(x[:2], x[2:], tau)])
    x = np.concatenate([q, qd])
    k1 = f(x); k2 = f(x + dt/2*k1); k3 = f(x + dt/2*k2); k4 = f(x + dt*k3)
    x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return x[:2], x[2:]

def quintic(q0, qf, T, t):
    s = np.clip(t / T, 0, 1)
    p = 10*s**3 - 15*s**4 + 6*s**5
    dp = (30*s**2 - 60*s**3 + 30*s**4) / T
    ddp = (60*s - 180*s**2 + 120*s**3) / T**2
    d = np.asarray(qf) - np.asarray(q0)
    return q0 + d*p, d*dp, d*ddp
