"""Dynamical rate model with neuromodulation — the whole-brain default.

This is the model the rest of Enactome is built around. Each neuron (or population) carries a
continuous firing rate r that evolves by a leaky ODE toward a gain-modulated input drive:

    tau * dr/dt = -r + phi( gain * (W_signed @ r + I_ext) + bias )

  - `W_signed` is the connectome synapse-count matrix with each presynaptic column multiplied
    by its neurotransmitter sign (ACh +1, GABA/GLUT -1); this is the SAME signed weight the
    static analyses used, now embedded in a dynamical system.
  - `phi` is a saturating nonlinearity (default: rectified tanh) so rates stay bounded.
  - **Neuromodulation** enters as a per-neuron multiplicative `gain` and additive `bias`,
    NOT as fast synaptic current. This matches the biology: dopamine, octopamine, serotonin
    act on a slow timescale to reconfigure the *transfer function* of their targets rather
    than to drive spikes directly. `NeuromodState` below computes gain/bias from modulator
    levels, so the neuromodulation model is consistent with the rate model by construction.

The model is deliberately simple enough to integrate thousands of neurons for seconds in
milliseconds, while capturing the computational property that matters at the circuit level:
gain-controlled, sign-respecting recurrent dynamics with a fixed point that the connectome
shapes.
"""
from __future__ import annotations
import numpy as np
from .backend import get_backend


# how each modulator maps onto (gain multiplier, bias add) of its receptor-bearing targets.
# Signs follow the standard neuromodulator roles; magnitudes are model parameters.
MODULATOR_EFFECT = {
    "dopamine":    {"gain": +0.6, "bias": +0.0},   # DA: gain up (incentive salience / vigor)
    "octopamine":  {"gain": +0.5, "bias": +0.1},   # OA (insect NE): arousal, gain + drive up
    "serotonin":   {"gain": -0.3, "bias": +0.0},   # 5-HT: gain down (behavioral restraint)
    "acetylcholine": {"gain": +0.3, "bias": +0.0}, # ACh (muscarinic modulatory pool): attention
}


class NeuromodState:
    """Holds modulator levels and turns them into per-neuron gain/bias for the rate model.

    receptor_frac[m] is a (n_neurons,) vector in [0,1]: how strongly each neuron expresses
    receptors for modulator m (e.g. derived from the atlas NT/receptor panels). level[m] is a
    scalar in [0,1] (e.g. set by a 'drug' input in the UI). The effective gain/bias are

        gain = 1 + Σ_m level[m] * receptor_frac[m] * MODULATOR_EFFECT[m]['gain']
        bias =     Σ_m level[m] * receptor_frac[m] * MODULATOR_EFFECT[m]['bias']
    """

    def __init__(self, n_neurons, receptor_frac=None):
        self.n = n_neurons
        self.receptor_frac = receptor_frac or {}
        self.level = {m: 0.0 for m in MODULATOR_EFFECT}

    def set_level(self, modulator, level):
        if modulator not in MODULATOR_EFFECT:
            raise KeyError(f"unknown modulator {modulator!r}")
        self.level[modulator] = float(level)

    def gain_bias(self):
        gain = np.ones(self.n, dtype=np.float32)
        bias = np.zeros(self.n, dtype=np.float32)
        for m, eff in MODULATOR_EFFECT.items():
            lvl = self.level.get(m, 0.0)
            if lvl == 0.0:
                continue
            rf = self.receptor_frac.get(m)
            rf = np.ones(self.n, dtype=np.float32) if rf is None else np.asarray(rf, np.float32)
            gain += lvl * rf * eff["gain"]
            bias += lvl * rf * eff["bias"]
        return gain, bias


class RateModel:
    """Recurrent rate network integrated with forward Euler.

    Parameters
    ----------
    W_signed : (n, n) signed weight matrix (row=post, col=pre) — connectome * NT sign.
    tau      : membrane/rate time constant (s).
    phi      : 'reltanh' (rectified tanh, bounded) or 'relu'.
    neuromod : optional NeuromodState (per-neuron gain/bias).
    prefer_gpu : use torch/CUDA if available.
    """

    def __init__(self, W_signed, tau=0.02, phi="reltanh", neuromod=None,
                 r_max=1.0, prefer_gpu=False):
        self.be = get_backend(prefer_gpu=prefer_gpu)
        self.n = W_signed.shape[0]
        self.tau = tau
        self.phi_name = phi
        self.r_max = r_max
        self.neuromod = neuromod
        # keep sparse weights sparse (whole-brain: 158k neurons dense is infeasible).
        import scipy.sparse as sp
        self.sparse = sp.issparse(W_signed)
        if self.sparse:
            self._Wsp = W_signed.tocsr().astype(np.float32)
            if self.be.kind == "torch":
                coo = self._Wsp.tocoo()
                idx = self.be.torch.tensor(np.vstack([coo.row, coo.col]), dtype=self.be.torch.long,
                                           device=self.be.device)
                val = self.be.torch.tensor(coo.data, dtype=self.be._dtype, device=self.be.device)
                self.W = self.be.torch.sparse_coo_tensor(idx, val, self._Wsp.shape).coalesce()
            else:
                self.W = self._Wsp
        else:
            self.W = self.be.array(W_signed)

    def _matmul(self, r):
        if self.sparse and self.be.kind == "numpy":
            return self._Wsp @ r
        if self.sparse and self.be.kind == "torch":
            return self.be.torch.sparse.mm(self.W, r.unsqueeze(1)).squeeze(1)
        return self.be.matmul(self.W, r)

    def _phi(self, x):
        be = self.be
        if self.phi_name == "relu":
            return be.maximum(x, 0.0)
        # rectified tanh: bounded in [0, r_max]
        return self.r_max * be.maximum(be.tanh(x), 0.0)

    def simulate(self, I_ext, dt=0.001, T=0.5, r0=None, record_every=1):
        """Integrate for T seconds. I_ext: (n,) constant external drive (or callable t->(n,)).
        Returns dict with final rate, time trace (subsampled), and time axis (all numpy)."""
        be = self.be
        n_steps = int(T / dt)
        r = be.zeros(self.n) if r0 is None else be.array(r0)
        if self.neuromod is not None:
            g, b = self.neuromod.gain_bias()
            gain, bias = be.array(g), be.array(b)
        else:
            gain, bias = 1.0, 0.0
        constant = not callable(I_ext)
        I = be.array(I_ext) if constant else None
        trace, times = [], []
        for step in range(n_steps):
            It = I if constant else be.array(I_ext(step * dt))
            drive = gain * (self._matmul(r) + It) + bias
            r = r + (dt / self.tau) * (-r + self._phi(drive))
            if step % record_every == 0:
                trace.append(be.to_numpy(r).copy())
                times.append(step * dt)
        return {"r_final": be.to_numpy(r), "trace": np.array(trace),
                "t": np.array(times), "backend": self.be.kind, "device": self.be.device}
