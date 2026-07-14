"""4-quadrant behavioral arena: MB valence + CX heading -> spatial navigation.

The canonical Enactome demo. Combines the connectome mushroom-body model with a behavioral
locomotion controller into one navigating agent:

  - **Heading** — each fly's heading is a behavioral angular state variable integrated over
    time (a reorientation applies a random turn, a run applies small jitter). It is NOT a
    neural model. The real BANC central-complex heading circuit is analyzed separately in
    experiments.py (cx_ring_architecture, cx_ring_topology, cx_heading_bump,
    cx_discrete_attractor), where every weight is a BANC synapse count. The `RingAttractor`
    classes below are a synthetic reference model kept only for comparison, not used here.
  - **MB (mushroom body)** — when a fly sits in a lit quadrant, the chosen optogenetic
    perturbation (via `mb_behavior.MBModel`) produces a valence signal (approach/avoid).
  - **Locomotion controller** — the standard klinokinesis rule: reorientation probability
    rises when valence *worsens* (temporal derivative). Aversive light -> bounce out ->
    net avoidance; appetitive light -> dwell -> net attraction.

Output is the **preference index** PI in [-1, +1]: +1 = spends all time in lit quadrants,
-1 = fully avoids them, 0 = indifferent. Aso et al. 2014 (eLife 04580) established that
optogenetic MBON activation induces cell-type-dependent attraction or avoidance; PI is the
spatial-preference readout used to quantify that effect in optogenetic place-preference
assays.
"""
from __future__ import annotations
import numpy as np


class RingAttractor:
    """Synthetic reference ring attractor (hand-wired von Mises kernel, NOT from the connectome).

    Kept only as an idealized comparison for the connectome-derived heading analysis in
    experiments.py. Its weight matrix is an analytic local-excitation/global-inhibition kernel
    with hand-chosen parameters; it does not use any BANC synapse counts. Do not present its
    output as a connectome result.
    """

    def __init__(self, n: int = 36, exc: float = 1.0, sigma: float = 0.6,
                 inh: float = 0.25, tau: float = 0.3):
        self.n = n
        self.theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        d = self.theta[:, None] - self.theta[None, :]
        # local excitation (von Mises kernel) minus uniform global inhibition (Delta7)
        self.W = exc * np.exp((np.cos(d) - 1) / sigma ** 2) - inh
        self.tau = tau
        self.a = np.exp((np.cos(self.theta) - 1) / sigma ** 2)  # initial bump at heading 0

    def step(self, ang_vel: float = 0.0, noise: float = 0.0, rng=None):
        rec = np.maximum(self.W @ self.a, 0.0)
        grad = np.gradient(self.a)
        self.a = self.a + self.tau * (-self.a + rec) - ang_vel * grad
        if noise and rng is not None:
            self.a = self.a + noise * rng.standard_normal(self.n)
        self.a = np.maximum(self.a, 0.0)
        s = self.a.sum()
        if s > 0:
            self.a *= self.n * 0.1 / s  # soft normalization keeps the bump alive

    def heading(self) -> float:
        return float(np.angle(np.sum(self.a * np.exp(1j * self.theta))))


class PopRingAttractor:
    """Vectorized bank of synthetic reference ring attractors (NOT from the connectome).

    Same hand-wired von Mises kernel as RingAttractor, run in parallel. Kept only as an
    idealized comparison; not used by the arena or by any connectome experiment.
    """

    def __init__(self, n_flies, n=24, exc=1.0, sigma=0.6, inh=0.25, tau=0.35, seed=0):
        self.n = n
        self.theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        d = self.theta[:, None] - self.theta[None, :]
        self.W = exc * np.exp((np.cos(d) - 1) / sigma ** 2) - inh   # (n, n)
        self.tau = tau
        rng = np.random.default_rng(seed)
        # initialize each fly's bump at a random heading
        h0 = rng.uniform(0, 2 * np.pi, n_flies)
        self.a = np.exp((np.cos(self.theta[None, :] - h0[:, None]) - 1) / sigma ** 2)  # (n_flies, n)

    def step(self, ang_vel):
        """ang_vel: (n_flies,) angular velocity input (rad/step) this timestep."""
        rec = np.maximum(self.a @ self.W.T, 0.0)                   # (n_flies, n)
        grad = np.gradient(self.a, axis=1)
        self.a = self.a + self.tau * (-self.a + rec) - ang_vel[:, None] * grad
        self.a = np.maximum(self.a, 0.0)
        s = self.a.sum(axis=1, keepdims=True)
        np.divide(self.a, s, out=self.a, where=s > 0)
        self.a *= self.n * 0.1

    def heading(self):
        """Population-vector heading of each fly's bump: (n_flies,)."""
        z = self.a @ np.exp(1j * self.theta)
        return np.angle(z)


def _quadrant(pos):
    a = np.arctan2(pos[:, 1], pos[:, 0]) % (2 * np.pi)
    return (a // (np.pi / 2)).astype(int)


def run_arena(valence_in_light: float, n_flies: int = 200, steps: int = 2000,
              speed: float = 0.012, p0: float = 0.12, k: float = 4.0, kv: float = 0.20,
              lit=(0, 2), seed: int = 0) -> dict:
    """Simulate `n_flies` in a circular 4-quadrant arena; opposite quadrants `lit` are ON.

    valence_in_light: the MB locomotion drive experienced while in a lit quadrant
        (negative = aversive perturbation, positive = appetitive). Get it from
        MBModel.activate_mbon(...) so the arena is connectome-driven.
    Returns preference index PI, per-fly light fraction, occupancy trace, final positions.
    """
    rng = np.random.default_rng(seed)
    R = 1.0
    th0 = rng.uniform(0, 2 * np.pi, n_flies)
    rr0 = np.sqrt(rng.uniform(0, 1, n_flies)) * R * 0.9
    pos = np.c_[rr0 * np.cos(th0), rr0 * np.sin(th0)]
    # heading is a per-fly angular state variable (behavioral heading integrator, no assumed
    # neural connectivity); reorientations add angular velocity, runs add small jitter
    head = rng.uniform(0, 2 * np.pi, n_flies)

    def in_light(p):
        return np.isin(_quadrant(p), lit)

    v_prev = np.where(in_light(pos), valence_in_light, 0.0)
    time_in_light = np.zeros(n_flies)
    occ = []
    for t in range(steps):
        il = in_light(pos)
        v_now = np.where(il, valence_in_light, 0.0)
        dv = v_now - v_prev                                   # >0 improving, <0 worsening
        # klinokinesis: reorient more when valence WORSENS (derivative) and while the current
        # location is aversive (level). Both terms are standard in insect place-avoidance.
        p_turn = np.clip(p0 - k * dv - kv * v_now, 0.02, 0.95)
        turn = rng.random(n_flies) < p_turn
        # klinokinesis output: a reorientation applies a large random turn, a run applies only
        # small heading jitter. This is a behavioral heading integrator, not a neural model.
        ang_vel = turn * rng.uniform(-np.pi, np.pi, n_flies) + (~turn) * rng.normal(0, 0.08, n_flies)
        head = (head + ang_vel) % (2 * np.pi)
        newpos = pos + speed * np.c_[np.cos(head), np.sin(head)]
        out = np.linalg.norm(newpos, axis=1) > R
        newpos[out] = pos[out]                                # stop at wall (bump unchanged)
        pos = newpos
        v_prev = v_now
        time_in_light += il
        if t % 40 == 0:
            occ.append(float(in_light(pos).mean()))
    frac_light = time_in_light / steps
    PI = float(2 * (frac_light.mean() - 0.5))                 # +prefers light / -avoids light
    return {"PI": PI, "frac_light": frac_light.tolist(), "occupancy": occ,
            "final_pos": pos.tolist(), "valence_in_light": valence_in_light}
