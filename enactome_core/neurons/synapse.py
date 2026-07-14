"""Synapse models calibrated to *Drosophila* central-neuron scale.

Fly central neurons are electrically tiny and compact, so synaptic parameters differ by orders
of magnitude from mammalian cortex. The values here are taken from the whole-brain connectome
LIF model of Shiu et al. 2023 (biorxiv 2023.05.02.539144), which in turn draws its single-
neuron constants from Kakaria & de Bivort 2017 and its synapse kinetics from Jürgensen et al.
2021. Using these means a Enactome spiking simulation is parameterised at the same scale as the
published fly whole-brain model, not with borrowed cortical numbers.

Canonical fly parameters (Shiu et al. 2023, Methods):
    V_rest = V_reset = -52 mV      resting / post-spike reset potential
    V_th   = -45 mV                spike threshold  (=> 7 mV to threshold from rest)
    R_m    = 10 MΩ                 membrane resistance (fly neurons are high-impedance)
    C_m    = 2 nF                  membrane capacitance
    tau_m  = R_m * C_m = 20 ms     membrane time constant
    tau_syn= 5 ms                  synaptic decay (alpha/exponential synapse)
    t_delay= 1.8 ms                axonal/synaptic transmission delay
    t_ref  = 2.2 ms                absolute refractory period
    w_syn  = 0.275 mV              depolarisation contributed by ONE synapse

The last value is the key scale bridge: the connectome gives an integer synapse count n_ij
between neurons i and j, and the postsynaptic voltage contribution of that connection is
n_ij * w_syn (with sign set by the presynaptic transmitter). ~25 simultaneous synapses from
rest (-52 mV) therefore reach threshold (-45 mV): 7 mV / 0.275 mV ≈ 25 synapses.
"""
from __future__ import annotations
import numpy as np

# --- canonical fly single-neuron + synapse constants (Shiu et al. 2023) ---
FLY_PARAMS = {
    "V_rest_mV": -52.0,
    "V_reset_mV": -52.0,
    "V_th_mV": -45.0,
    "R_m_Mohm": 10.0,
    "C_m_nF": 2.0,
    "tau_m_ms": 20.0,      # R_m * C_m
    "tau_syn_ms": 5.0,
    "t_delay_ms": 1.8,
    "t_ref_ms": 2.2,
    "w_syn_mV": 0.275,     # per-synapse postsynaptic depolarisation
}

# threshold distance and implied synapse count to spike (sanity anchor)
SYN_TO_THRESHOLD = (FLY_PARAMS["V_th_mV"] - FLY_PARAMS["V_rest_mV"]) / FLY_PARAMS["w_syn_mV"]  # ~25.5


class ExpSynapse:
    """Exponential (current/voltage) synapse: g decays with tau_syn; each presynaptic spike
    adds n_synapses * w_syn * sign to the postsynaptic drive. This is the synapse used by the
    fly whole-brain LIF (Shiu et al.): dg/dt = -g/tau_syn, v receives g.

    Parameters
    ----------
    W_counts : (n_post, n_pre) integer/float connectome synapse-count matrix.
    nt_sign  : (n_pre,) presynaptic transmitter sign (+1 ACh, -1 GABA/GLUT, 0 modulatory).
    w_syn_mV, tau_syn_ms : fly-calibrated defaults from FLY_PARAMS.
    """

    def __init__(self, W_counts, nt_sign, w_syn_mV=FLY_PARAMS["w_syn_mV"],
                 tau_syn_ms=FLY_PARAMS["tau_syn_ms"]):
        W = np.asarray(W_counts, dtype=np.float32)
        s = np.asarray(nt_sign, dtype=np.float32)
        # signed voltage-weight matrix in mV per presynaptic spike
        self.W_mV = (W * s[None, :]) * w_syn_mV
        self.tau_syn = tau_syn_ms * 1e-3       # s
        self.n_post, self.n_pre = W.shape

    def kick(self, g, spikes):
        """Add the voltage kick from presynaptic `spikes` (n_pre,) to conductance state g."""
        return g + self.W_mV @ np.asarray(spikes, dtype=np.float32)

    def decay(self, g, dt):
        return g - dt * g / self.tau_syn

    def signed_weight_matrix(self):
        """Return the signed per-spike voltage weight matrix (mV), for use by LIFNetwork."""
        return self.W_mV


def as_lif_kwargs(prefer_gpu=False):
    """Return LIFNetwork constructor kwargs in fly units (mV, s), from FLY_PARAMS.

    The LIF network works in volts internally; we pass rest=0 reference by shifting so
    V_th - V_rest = 7 mV becomes the threshold, and w_syn is already in mV.
    """
    p = FLY_PARAMS
    return dict(
        tau_m=p["tau_m_ms"] * 1e-3,
        tau_syn=p["tau_syn_ms"] * 1e-3,
        V_th=(p["V_th_mV"] - p["V_rest_mV"]),   # 7 mV above rest
        V_reset=0.0,                             # rest reference
        E_L=0.0,
        R=1.0,                                   # weights already carry mV
        t_ref=p["t_ref_ms"] * 1e-3,
        w_scale=p["w_syn_mV"],                   # per-synapse mV
        prefer_gpu=prefer_gpu,
    )
