"""Spiking neuron models: leaky integrate-and-fire (LIF/IAF) and Hodgkin-Huxley (HH).

These are the higher-fidelity tiers the UI can switch to. They share the CPU/GPU `Backend`
with the rate model, and take the SAME signed connectome weight matrix, so a circuit built at
the rate level can be re-simulated spike-by-spike without rewiring. Synaptic coupling here is
a simple current injection proportional to presynaptic spikes filtered by an exponential
synapse — enough to run illustrative experiments, not a full conductance-based synapse.
"""
from __future__ import annotations
import numpy as np
from .backend import get_backend


class LIFNetwork:
    """Leaky integrate-and-fire network (current-based exponential synapses).

    tau_m dV/dt = -(V - E_L) + R*I_syn + R*I_ext ; spike when V>=V_th, reset to V_reset,
    refractory for t_ref. I_syn is an exponentially-filtered sum of signed presynaptic spikes.
    """

    def __init__(self, W_signed, tau_m=0.02, tau_syn=0.005, V_th=1.0, V_reset=0.0,
                 E_L=0.0, R=1.0, t_ref=0.002, w_scale=0.01, prefer_gpu=False):
        self.be = get_backend(prefer_gpu=prefer_gpu)
        self.n = W_signed.shape[0]
        import scipy.sparse as sp
        self.sparse = sp.issparse(W_signed)
        if self.sparse:
            # keep sparse on CPU (numpy path); LIF over a large sparse graph stays feasible
            self._Wsp = (W_signed * w_scale).tocsr().astype(np.float32)
            self.W = None
        else:
            self.W = self.be.array(W_signed * w_scale)
        self.tau_m, self.tau_syn = tau_m, tau_syn
        self.V_th, self.V_reset, self.E_L, self.R = V_th, V_reset, E_L, R
        self.t_ref = t_ref

    def simulate(self, I_ext, dt=0.0005, T=0.5, seed=0):
        be = self.be
        n_steps = int(T / dt)
        V = be.array(np.full(self.n, self.E_L))
        g = be.zeros(self.n)                      # synaptic current state
        ref = be.zeros(self.n)                    # refractory countdown (s)
        I = be.array(I_ext) if not callable(I_ext) else None
        spike_counts = be.zeros(self.n)
        raster_t, raster_i = [], []
        for step in range(n_steps):
            It = I if I is not None else be.array(I_ext(step * dt))
            g = g + dt * (-g / self.tau_syn)
            not_ref = ref <= 0.0
            dV = (-(V - self.E_L) + self.R * g + self.R * It) * (dt / self.tau_m)
            V = be.where(not_ref, V + dV, V)
            spikes = (V >= self.V_th) & not_ref
            sp = be.array(spikes.astype(np.float32)) if be.kind == "numpy" else spikes.float()
            # propagate spikes through signed weights into synaptic current
            if self.sparse:
                g = g + self._Wsp @ be.to_numpy(sp)
            else:
                g = g + be.matmul(self.W, sp)
            V = be.where(spikes, be.array(np.full(self.n, self.V_reset)) if be.kind=="numpy"
                         else be.torch.full((self.n,), self.V_reset, device=self.be.device), V)
            ref = be.where(spikes, be.array(np.full(self.n, self.t_ref)) if be.kind=="numpy"
                           else be.torch.full((self.n,), self.t_ref, device=self.be.device), ref - dt)
            spike_counts = spike_counts + sp
            idx = np.where(be.to_numpy(sp) > 0)[0]
            raster_t.extend([step * dt] * len(idx)); raster_i.extend(idx.tolist())
        rates = be.to_numpy(spike_counts) / T
        return {"rates_hz": rates, "raster_t": np.array(raster_t), "raster_i": np.array(raster_i),
                "backend": self.be.kind, "device": self.be.device}


class HHNeuron:
    """Hodgkin-Huxley point neuron(s) (classic Na/K/leak), vectorized over a population.

    Fast, standard formulation (Traub-Miles style rate constants) run with forward Euler at
    small dt. Provided so Enactome can run *sample* biophysical experiments (e.g. an f-I curve
    or a single-cell trace) on CPU or GPU; it is not wired into the whole-brain network by
    default because HH over 10^5 neurons is outside the "fast" regime this tool targets.
    """

    def __init__(self, n=1, C=1.0, gNa=120.0, gK=36.0, gL=0.3,
                 ENa=50.0, EK=-77.0, EL=-54.4, prefer_gpu=False):
        self.be = get_backend(prefer_gpu=prefer_gpu)
        self.n = n
        self.C, self.gNa, self.gK, self.gL = C, gNa, gK, gL
        self.ENa, self.EK, self.EL = ENa, EK, EL

    @staticmethod
    def _alpha_beta(V):
        # rate constants (V in mV); guards for the removable singularities at V=-40,-55
        import numpy as _np
        am = 0.1 * (V + 40) / (1 - _np.exp(-(V + 40) / 10) + 1e-9)
        bm = 4.0 * _np.exp(-(V + 65) / 18)
        ah = 0.07 * _np.exp(-(V + 65) / 20)
        bh = 1.0 / (1 + _np.exp(-(V + 35) / 10))
        an = 0.01 * (V + 55) / (1 - _np.exp(-(V + 55) / 10) + 1e-9)
        bn = 0.125 * _np.exp(-(V + 65) / 80)
        return am, bm, ah, bh, an, bn

    def simulate(self, I_ext, dt=0.01, T=50.0):
        """I_ext: scalar or (n,) injected current (uA/cm^2). dt,T in ms. numpy path (HH is a
        sample/demonstration tier; kept on numpy for the transcendental rate constants)."""
        import numpy as np
        n_steps = int(T / dt)
        V = np.full(self.n, -65.0)
        m = 0.05 * np.ones(self.n); h = 0.6 * np.ones(self.n); ng = 0.32 * np.ones(self.n)
        I = np.full(self.n, I_ext) if np.isscalar(I_ext) else np.asarray(I_ext, float)
        Vtr = np.zeros((n_steps, self.n)); spikes = np.zeros(self.n)
        prevV = V.copy()
        for s in range(n_steps):
            am, bm, ah, bh, an, bn = self._alpha_beta(V)
            m += dt * (am * (1 - m) - bm * m)
            h += dt * (ah * (1 - h) - bh * h)
            ng += dt * (an * (1 - ng) - bn * ng)
            INa = self.gNa * m**3 * h * (V - self.ENa)
            IK = self.gK * ng**4 * (V - self.EK)
            IL = self.gL * (V - self.EL)
            V = V + dt * (I - INa - IK - IL) / self.C
            spikes += ((prevV < 0) & (V >= 0)).astype(float)   # upward zero-crossing
            prevV = V.copy(); Vtr[s] = V
        return {"V": Vtr, "t": np.arange(n_steps) * dt, "spike_counts": spikes,
                "rate_hz": spikes / (T / 1000.0), "backend": self.be.kind}
