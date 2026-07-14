"""Mushroom-body learning + locomotion model — replicates Aso et al. 2014 (eLife 04580).

The MB is modeled as: sparse Kenyon-cell odor code -> KC->MBON synapses (plastic) ->
MBON ensemble -> a locomotion/action-selection layer that turns the MBON valence balance
into an approach/avoid drive.

Grounding in the connectome (BANC):
  - MBON behavioral valence is read from the MBON's predicted neurotransmitter, following
    Aso 2014's central finding: glutamatergic MBONs drive AVOIDANCE, GABAergic/cholinergic
    MBONs drive ATTRACTION. (val_sign below.)
  - DAN-gated plasticity: pairing an odor (KC activity) with DAN activity DEPRESSES the
    KC->MBON synapses of the co-active KCs in that DAN's compartment. Because the depressed
    MBON normally pushes the OPPOSITE valence to the DAN's teaching signal, depression shifts
    the ensemble balance toward the DAN's valence — i.e. a punishment DAN writes avoidance.

This module is connectome-driven but deliberately simple: rates, not spikes; a single
plastic layer; a linear locomotion readout. It is fast enough to run thousands of trials.
"""
from __future__ import annotations
import numpy as np

# MBON NT -> behavioral valence sign (Aso 2014). +1 = attraction/approach, -1 = avoidance.
VAL_SIGN = {"ACH": +1.0, "GABA": +1.0, "GLUT": -1.0, "DA": 0.0, "SER": 0.0}


def mbon_valence(nt_types) -> np.ndarray:
    """Per-MBON behavioral valence sign from predicted NT (Aso 2014 rule)."""
    return np.array([VAL_SIGN.get(str(nt), 0.0) for nt in nt_types])


class MBModel:
    """Connectome-constrained MB learning + locomotion model.

    Parameters
    ----------
    W_kc_mbon : (n_kc, n_mbon) baseline KC->MBON synapse-count matrix (excitatory).
    mbon_val  : (n_mbon,) behavioral valence sign per MBON (from mbon_valence()).
    dan_comp  : (n_dan, n_mbon) DAN->MBON compartment overlap (which MBONs a DAN modulates).
    dan_sign  : (n_dan,) teaching valence per DAN (+1 reward / -1 punishment).
    """

    def __init__(self, W_kc_mbon, mbon_val, dan_comp=None, dan_sign=None):
        self.W0 = W_kc_mbon.astype(float)
        self.W = self.W0.copy()
        self.mbon_val = np.asarray(mbon_val, float)
        self.dan_comp = dan_comp
        self.dan_sign = None if dan_sign is None else np.asarray(dan_sign, float)

    # --- odor code ---
    def odor_code(self, n_active=None, seed=0, n_odors=1):
        """Sparse binary KC activation pattern(s). ~5% of KCs active per odor (fly-realistic)."""
        rng = np.random.default_rng(seed)
        n_kc = self.W0.shape[0]
        k = n_active or max(1, int(0.05 * n_kc))
        codes = np.zeros((n_odors, n_kc))
        for i in range(n_odors):
            codes[i, rng.choice(n_kc, k, replace=False)] = 1.0
        return codes if n_odors > 1 else codes[0]

    # --- forward: odor -> MBON activity -> locomotion drive ---
    def mbon_activity(self, kc_vec):
        return np.maximum(kc_vec @ self.W, 0.0)

    def approach_drive(self, kc_vec):
        """Net locomotion valence: sum of MBON activity weighted by behavioral sign.

        >0 = net approach, <0 = net avoidance. This is the ensemble valence read-out.
        """
        a = self.mbon_activity(kc_vec)
        return float(a @ self.mbon_val)

    # --- optogenetic perturbation (Aso's activation experiment) ---
    def activate_mbon(self, mbon_idx, level=1.0):
        """Directly drive one or more MBONs; return the induced locomotion valence."""
        a = np.zeros(self.W.shape[1])
        a[np.atleast_1d(mbon_idx)] = level
        return float(a @ self.mbon_val)

    # --- DAN-gated plasticity (the learning rule) ---
    def train(self, kc_vec, dan_idx, rate=0.5):
        """Pair an odor (kc_vec) with DAN activation: depress co-active KC->MBON synapses
        in that DAN's compartment. Returns nothing; mutates self.W.
        """
        if self.dan_comp is None:
            raise ValueError("dan_comp required for training")
        # dan_idx may be a scalar or a list of DANs; union their compartments.
        comp = self.dan_comp[np.atleast_1d(dan_idx)]     # (n_sel, n_mbon)
        targets = comp.sum(axis=0) > 0                    # MBONs in any selected DAN's compartment
        active_kc = kc_vec > 0
        # multiplicative depression of active-KC -> compartment-MBON synapses
        depress = np.outer(active_kc, targets)
        self.W = self.W * (1.0 - rate * depress)

    def reset(self):
        self.W = self.W0.copy()


# --- Perturbation-hypothesis engine ------------------------------------------------
# Optogenetic / ablation driver genes available in the fly toolkit, per cell class.
# These are the GAL4/split-GAL4 handles a bench experiment would actually use.
PERTURBATION_HANDLES = {
    "mushroom_body_output_neuron": {
        "drivers": ["MB-MBON split-GAL4 lines (Aso 2014)", "VT-GAL4"],
        "activate": "UAS-CsChrimson (optogenetic depolarization)",
        "silence": "UAS-Kir2.1 / UAS-shibire[ts]",
        "ablate": "UAS-reaper / UAS-hid",
    },
    "mushroom_body_dopaminergic_neuron": {
        "drivers": ["TH-GAL4 (PPL1)", "R58E02-GAL4 (PAM)", "split-GAL4 DAN lines"],
        "activate": "UAS-CsChrimson",
        "silence": "UAS-Kir2.1",
        "ablate": "UAS-reaper",
    },
    "kenyon_cell": {
        "drivers": ["OK107-GAL4", "MB247-GAL4"],
        "activate": "UAS-CsChrimson",
        "silence": "UAS-Kir2.1 / UAS-shibire[ts]",
        "ablate": "hydroxyurea ablation / UAS-reaper",
    },
}


def perturb(model: "MBModel", mbon_idx=None, dan_idx=None, mode="activate",
            odor=None, level=1.0) -> dict:
    """Predict the behavioral consequence of a perturbation — a testable hypothesis.

    mode: 'activate' (drive the cell), 'silence'/'ablate' (remove its output).
    Returns baseline vs perturbed locomotion valence and a plain-language prediction.
    """
    if odor is None:
        odor = model.odor_code(seed=0)
    base = model.approach_drive(odor)

    if mbon_idx is not None:
        idxs = np.atleast_1d(mbon_idx)
        if mode == "activate":
            perturbed = base + model.activate_mbon(idxs, level)
        else:  # silence / ablate: zero these MBONs' contribution
            val = model.mbon_val.copy(); val[idxs] = 0.0
            perturbed = float(model.mbon_activity(odor) @ val)
        sign = model.mbon_val[idxs].mean()
        cls = "mushroom_body_output_neuron"
    elif dan_idx is not None:
        # DAN perturbation acts through learning: activate = write memory of DAN's sign
        model.reset()
        if mode == "activate":
            for _ in range(5):
                model.train(odor, dan_idx, rate=0.5)
            perturbed = model.approach_drive(odor)
        else:
            perturbed = base  # silencing a DAN blocks new learning; no acute change
        sign = model.dan_sign[np.atleast_1d(dan_idx)].mean() if model.dan_sign is not None else 0.0
        cls = "mushroom_body_dopaminergic_neuron"
        model.reset()
    else:
        raise ValueError("specify mbon_idx or dan_idx")

    delta = perturbed - base
    direction = "approach" if delta > 1e-6 else ("avoidance" if delta < -1e-6 else "no change")
    handles = PERTURBATION_HANDLES.get(cls, {})
    return {
        "mode": mode, "cell_class": cls,
        "baseline_drive": round(base, 3), "perturbed_drive": round(perturbed, 3),
        "delta": round(delta, 3), "predicted_behavior": direction,
        "genetic_handle": handles.get(mode, handles.get("activate")),
        "drivers": handles.get("drivers"),
        "hypothesis": f"{mode} of this {cls.replace('_',' ')} is predicted to bias behavior toward "
                      f"{direction} (Δ drive {delta:+.2f}); test with {handles.get(mode, '')}.",
    }
