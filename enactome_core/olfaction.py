"""Olfactory pathway + lateral horn: the INNATE valence channel of the fly brain.

Together with `mb_behavior` (the LEARNED channel) this implements the fundamental
architecture of the *Drosophila* olfactory brain:

    odorant ─▶ glomeruli ─▶ uniglomerular PNs ─┬─▶ lateral horn (LH)  ── INNATE valence
                                               │      (hardwired, evolutionarily fixed)
                                               └─▶ Kenyon cells ─▶ MB  ── LEARNED valence
                                                      (plastic, written by dopamine)

The two channels read the SAME projection-neuron code but compute different things:
  - **LH** applies a fixed read-out: each glomerulus carries an innate valence sign
    (from the DoOR/behavioural literature, Fig. S7 of the BANC olfactory PN paper), and
    LH output neurons inherit a valence bias from the glomeruli that converge on them.
    Nothing about the LH changes with experience.
  - **MB** starts valence-neutral and acquires odor→valence associations only when a
    dopaminergic teaching signal coincides with odor-driven KC activity (see
    `mb_behavior.MBModel`). This is the substrate of memory.

`FlyBrain` below wires both channels to one odor and returns the innate, learned, and
combined valence — enabling the double-dissociation test that defines the hypothesis:
lesion the LH → lose innate preference but keep learning; lesion the MB → keep innate
preference but lose memory.
"""
from __future__ import annotations
import numpy as np


# innate glomerular valence (Fig S7): appetitive/aversive/neutral -> sign
GLOM_VALENCE_SIGN = {"appetitive": +1.0, "aversive": -1.0,
                     "complex": 0.0, "disputed": 0.0, "unknown": 0.0}


class LateralHorn:
    """Innate valence read-out: glomerulus -> uPN -> LHON, with fixed valence weights."""

    def __init__(self, P, W_pn_lhon, glom_sign):
        """P: (n_glom, n_upn) glom->uPN. W_pn_lhon: (n_upn, n_lhon). glom_sign: (n_glom,)."""
        self.P = np.asarray(P, float)
        self.W = np.asarray(W_pn_lhon, float)
        self.glom_sign = np.asarray(glom_sign, float)
        # per-uPN innate sign inherited from its glomerulus
        self.upn_sign = self.glom_sign @ self.P
        # per-LHON valence index (synapse-weighted mean presynaptic glom sign)
        signed = self.upn_sign @ self.W
        total = self.W.sum(axis=0)
        self.lhon_vi = np.divide(signed, total, out=np.zeros_like(signed), where=total > 0)

    def pn_drive(self, glom_activation):
        """glomerular activation (n_glom,) -> uPN drive (n_upn,)."""
        return glom_activation @ self.P

    def innate_valence(self, glom_activation, lesion=False):
        """Net innate valence for an odor: the labeled-line valence carried by the activated
        projection neurons and pooled through the LH. Each uPN inherits its glomerulus's
        hardwired valence sign; the LH forwards their synapse-weighted sum onto LHONs.
        lesion=True zeros the LH (double-dissociation).
        """
        if lesion:
            return 0.0
        upn = self.pn_drive(glom_activation)          # per-uPN drive for this odor
        # valence-signed uPN drive, weighted by how strongly each uPN reaches the LH
        lh_reach = self.W.sum(axis=1)                 # total uPN->LHON synapses per uPN
        return float((upn * self.upn_sign * lh_reach).sum())


class FlyBrain:
    """The canonical fly-brain model: innate (LH) + learned (MB) olfactory valence.

    Parameters
    ----------
    lh   : LateralHorn (innate channel).
    mb   : mb_behavior.MBModel (learned channel).
    W_pn_kc : (n_upn, n_kc) uPN->KC, routes the SAME PN code into the MB.
    w_innate, w_learned : mixing weights for the combined behavioral drive.
    """

    def __init__(self, lh: "LateralHorn", mb, W_pn_kc, w_innate=1.0, w_learned=1.0):
        self.lh = lh
        self.mb = mb
        self.W_pn_kc = np.asarray(W_pn_kc, float)
        self.w_innate = w_innate
        self.w_learned = w_learned

    def kc_code(self, glom_activation, sparsity=0.05):
        """Route glomerular activation -> uPN -> KC, then apply MB sparse coding (top-k)."""
        upn = self.lh.pn_drive(glom_activation)
        raw = upn @ self.W_pn_kc
        k = max(1, int(sparsity * raw.size))
        thr = np.partition(raw, -k)[-k]
        return (raw >= thr).astype(float) * (raw > 0)

    def valence(self, glom_activation, lh_lesion=False, mb_lesion=False):
        """Return innate, learned, and combined valence for an odor."""
        innate = self.lh.innate_valence(glom_activation, lesion=lh_lesion)
        kc = self.kc_code(glom_activation)
        learned = 0.0 if mb_lesion else self.mb.approach_drive(kc)
        combined = self.w_innate * innate + self.w_learned * learned
        return {"innate": innate, "learned": learned, "combined": combined}

    def train_odor(self, glom_activation, dan_idx, rate=0.5, reps=5):
        """Form a memory: pair this odor's KC code with a DAN teaching signal (MB plasticity)."""
        kc = self.kc_code(glom_activation)
        for _ in range(reps):
            self.mb.train(kc, dan_idx, rate=rate)


# genetic handles for optogenetic/silencing perturbation of LH cell types
# Generic, well-established genetic tooling for targeting LH cell classes. These name standard
# reagent *types* (LHON-targeting split-GAL4 lines + standard effectors), not specific published
# driver collections, so no external citation is implied.
LH_PERTURBATION_HANDLES = {
    "lateral_horn_output_neuron": {"drivers": ["LHON-targeting split-GAL4 lines"],
                                   "silence": "UAS-Kir2.1 / UAS-shibire^ts",
                                   "activate": "UAS-CsChrimson"},
    "lateral_horn_local_neuron": {"drivers": ["LHLN-targeting split-GAL4 lines"],
                                  "silence": "UAS-Kir2.1", "activate": "UAS-CsChrimson"},
    "lateral_horn_centrifugal_neuron": {"drivers": ["LHCENT-targeting split-GAL4 lines"],
                                        "silence": "UAS-Kir2.1", "activate": "UAS-CsChrimson"},
}


def lh_perturbation_hypothesis(lh: "LateralHorn", lhon_valence_index, lhon_types,
                               target_type, mode="silence"):
    """Predict the innate-behavior consequence of perturbing a set of LHONs.

    Given per-LHON valence indices and a target cell-type string (matched as a prefix of the
    LHON Primary Cell Type), estimate how silencing or activating that group shifts the net
    innate valence read-out, and return a testable, genetically-grounded hypothesis.
    """
    import numpy as np
    vi = np.asarray(lhon_valence_index, float)
    types = np.asarray([str(t) for t in lhon_types])
    hit = np.array([t.startswith(target_type) for t in types])
    n_hit = int(hit.sum())
    if n_hit == 0:
        return {"target_type": target_type, "n_matched": 0,
                "hypothesis": f"No LHONs match type prefix {target_type!r}."}
    mean_vi = float(vi[hit].mean())
    # silencing removes these LHONs' contribution; the net innate readout shifts OPPOSITE to
    # their mean valence (silencing aversive-biased LHONs disinhibits approach, and vice-versa).
    sign = "silence" if mode == "silence" else "activate"
    if mode == "silence":
        predicted = "increased approach" if mean_vi < 0 else "increased avoidance"
    else:
        predicted = "increased avoidance" if mean_vi < 0 else "increased approach"
    handle = LH_PERTURBATION_HANDLES.get("lateral_horn_output_neuron", {})
    return {"target_type": target_type, "n_matched": n_hit,
            "mean_valence_index": round(mean_vi, 3),
            "mode": sign, "predicted_behavior_shift": predicted,
            "genetic_handle": {"drivers": handle.get("drivers"),
                               "effector": handle.get(mode, handle.get("silence"))},
            "hypothesis": (f"{'Silencing' if mode=='silence' else 'Activating'} {n_hit} {target_type}-type LHONs "
                           f"(mean valence index {mean_vi:+.2f}) is predicted to cause "
                           f"{predicted} to odors that recruit them — testable with "
                           f"{handle.get('drivers', ['LH split-GAL4'])[0]} + "
                           f"{handle.get(mode, 'UAS-Kir2.1')}.")}


def load_flybrain(data_dir):
    """Build a FlyBrain from the shipped sparse connectome bundle (data/fb_*.npz).

    Returns (brain, meta) where meta carries glom names, MBON/DAN types for lookups.
    """
    import os
    import numpy as np
    from scipy import sparse
    from . import mb_behavior as mbb

    def L(name):
        return sparse.load_npz(os.path.join(data_dir, name)).toarray()

    W_pn_lhon = L("fb_W_pn_lhon.npz")
    W_pn_kc = L("fb_W_pn_kc.npz")
    W_kc_mbon = L("fb_W_kc_mbon.npz")
    dan_comp = L("fb_dan_comp.npz")
    m = np.load(os.path.join(data_dir, "fb_meta.npz"), allow_pickle=True)
    lh = LateralHorn(m["P"], W_pn_lhon, m["gval"])
    mb = mbb.MBModel(W_kc_mbon, m["mbon_val"], dan_comp, m["dan_sign"])
    brain = FlyBrain(lh, mb, W_pn_kc)
    meta = {"gloms": list(m["gloms"]), "mbon_nt": list(m["mbon_nt"]),
            "mbon_types": list(m["mbon_types"]), "dan_types": list(m["dan_types"]),
            "gval": m["gval"]}
    if "lhon_types" in m.files:
        meta["lhon_types"] = list(m["lhon_types"])
        meta["lhon_vi"] = m["lhon_vi"]
    return brain, meta
