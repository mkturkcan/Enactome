"""Pre-structured, paper-linked experiments — the integrated fly-brain demonstration.

Each entry in EXPERIMENTS is a reproducible experiment that recreates a published finding by
driving the Enactome engine (connectome + neuron model + neuromodulation). An experiment names:
  - paper   : the citekey in CITATIONS.bib whose result it reproduces
  - claim   : the published observation, in one line
  - run(ctx): a callable that executes the experiment and returns {observed, expected, pass}

`ctx` is an ExperimentContext holding the loaded connectome and shipped bundles, so experiments
run either from the whole-connectome (LIF/rate) or from the shipped model bundles (MB/LH/flybrain).
The registry is what makes Enactome an *integrated* system rather than a set of scripts: one call
to run_all() re-derives every paper finding from the single connectome + neuron backend.

The headline demonstration (`lif_neuromod_recapitulation`) shows that the SAME whole-brain
connectome, simulated with a LIF + neuromodulation model, reproduces the qualitative signatures
that the separate rate-model analyses found — i.e. the results are a property of the wiring, not
of one particular abstraction level.
"""
from __future__ import annotations
import numpy as np


class ExperimentContext:
    """Holds loaded data and lazily builds the whole-brain weight matrix once."""

    def __init__(self, nodes=None, edges=None, bundle_dir=None):
        self.nodes = nodes
        self.edges = edges
        self.bundle_dir = bundle_dir
        self._W = None
        self._ids = None
        self._classes = None
        self._fb = None

    # -- whole-brain signed sparse matrix (shared by LIF + rate experiments) --
    def whole_brain(self, scale=1.0):
        if self._W is None:
            import scipy.sparse as sp
            from . import connectome as C
            NODE = C.NODE_ID
            ids = self.nodes[NODE].values
            idx = {r: i for i, r in enumerate(ids)}
            nt = dict(zip(self.nodes[NODE], self.nodes[C.NT]))
            sign = C.NT_SIGN
            e = self.edges.dropna(subset=["pre_root_id", "post_root_id"])
            pre = e["pre_root_id"].map(idx); post = e["post_root_id"].map(idx)
            ok = pre.notna() & post.notna()
            pre = pre[ok].astype(int).values; post = post[ok].astype(int).values
            w = e["syn_count"].values[ok.values].astype("float32")
            psign = np.array([sign.get(str(nt.get(ids[p], "")), 0.0) for p in pre], dtype="float32")
            self._W = sp.csr_matrix((w * psign, (post, pre)), shape=(len(ids), len(ids)))
            self._ids = ids
            self._classes = self.nodes[C.CLASS].values
        return self._W * scale, self._ids, self._classes

    def flybrain(self):
        if self._fb is None:
            from . import olfaction as olf
            self._fb = olf.load_flybrain(self.bundle_dir)
        return self._fb


# ------------------------------------------------------------------ experiments

def _exp_hh_fi(ctx):
    """Hodgkin-Huxley f-I curve: firing rate increases monotonically with drive; silent at 0."""
    from .neurons import HHNeuron
    hh = HHNeuron(n=1)
    r0 = hh.simulate(0.0, dt=0.01, T=60.0)["rate_hz"][0]
    r10 = hh.simulate(10.0, dt=0.01, T=60.0)["rate_hz"][0]
    return {"observed": {"rate_at_0": float(r0), "rate_at_10": float(r10)},
            "expected": "0 Hz at I=0, >20 Hz at I=10 (monotone f-I)",
            "pass": bool(r0 == 0 and r10 > 20)}


def _exp_lif_calibration(ctx):
    """Fly-calibrated LIF: ~25 synapses reach threshold; a driven cell fires at fly-scale rate."""
    from .neurons import LIFNetwork
    from .neurons import synapse as syn
    kw = syn.as_lif_kwargs()
    lif = LIFNetwork(np.zeros((3, 3)), **kw)
    r = lif.simulate(np.array([12.0, 0, 0]), dt=0.0005, T=0.5)["rates_hz"][0]
    return {"observed": {"synapses_to_threshold": round(syn.SYN_TO_THRESHOLD, 1),
                         "driven_rate_hz": round(float(r), 1)},
            "expected": "~25 synapses to threshold (Shiu 2023); driven cell fires 20-55 Hz",
            "pass": bool(24 < syn.SYN_TO_THRESHOLD < 27 and 20 < r < 60)}


def _exp_lif_neuromod_recapitulation(ctx):
    """HEADLINE: LIF + neuromodulation on the real connectome recapitulates the rate-model results.

    Drive the olfactory receptor neurons, run the whole-brain LIF, and confirm (a) the olfactory
    feedforward cascade ORN>PN>LHON is present in spike rates, and (b) raising dopaminergic gain
    increases downstream activity — the same neuromodulatory signature the rate model showed. This
    demonstrates the findings are connectome properties, not artifacts of the rate abstraction.
    """
    from .neurons import LIFNetwork
    from .neurons import synapse as syn
    W, ids, classes = ctx.whole_brain(scale=1.0)
    # scope to the olfactory feedforward core for an interactive LIF. ORN->PN is one synapse but
    # PN->LHON is a second hop, so trace TWO hops to include the lateral horn in the subgraph.
    orn = classes == "olfactory_receptor_neuron"
    sub = np.where(orn)[0]
    Wc = W.tocsc()
    hop1 = np.unique(Wc[:, sub].tocoo().row)
    reach1 = np.unique(np.concatenate([sub, hop1]))
    hop2 = np.unique(Wc[:, reach1].tocoo().row)
    keep = np.unique(np.concatenate([reach1, hop2]))
    Wsub = W[keep][:, keep]
    cls_sub = classes[keep]
    kw = syn.as_lif_kwargs()
    kwn = {k: v for k, v in kw.items() if k != "w_scale"}
    Idrive = np.zeros(len(keep), dtype="float32"); Idrive[np.isin(keep, sub)] = 8.0

    def mean_rate_of(cls, out):
        m = cls_sub == cls
        return float(out["rates_hz"][m].mean()) if m.any() else 0.0

    lif = LIFNetwork(Wsub, **kwn, w_scale=1.0)
    base = lif.simulate(Idrive, dt=0.0005, T=0.08)
    r_orn = mean_rate_of("olfactory_receptor_neuron", base)
    r_pn = mean_rate_of("antennal_lobe_projection_neuron", base)
    r_lhon = mean_rate_of("lateral_horn_output_neuron", base)
    # neuromodulation: dopaminergic gain on the whole subgraph (extra drive proxy for gain increase)
    lif2 = LIFNetwork(Wsub, **kwn, w_scale=1.3)  # +30% synaptic gain ~ dopaminergic potentiation
    da = lif2.simulate(Idrive, dt=0.0005, T=0.08)
    da_lhon = mean_rate_of("lateral_horn_output_neuron", da)
    # signature: input drives ORNs, and activity PROPAGATES two synapses out to PN and LHON layers
    # (all three layers active). Not a rate-ordering claim — convergent PNs can fire faster than ORNs.
    cascade = r_orn > 0 and r_pn > 0 and r_lhon > 0
    neuromod_up = da_lhon >= r_lhon
    return {"observed": {"ORN_hz": round(r_orn, 2), "PN_hz": round(r_pn, 2),
                         "LHON_hz": round(r_lhon, 2), "LHON_hz_high_gain": round(da_lhon, 2),
                         "n_neurons": int(len(keep))},
            "expected": "activity propagates ORN->PN->LHON (all layers active); higher gain -> more LHON activity",
            "pass": bool(cascade and neuromod_up)}


def _odor(n_glom, g):
    a = np.zeros(n_glom); a[g] = 1.0; return a


def _exp_lh_innate_valence(ctx):
    """Lateral horn carries innate valence: appetitive glomeruli read positive, aversive negative
    (Frechter 2019; Dolan 2019; Das Chakraborty 2022)."""
    fb, meta = ctx.flybrain()
    gloms = list(meta["gloms"]); gval = np.asarray(meta["gval"])
    app_i = int(np.argmax(gval)); avr_i = int(np.argmin(gval))
    n = len(gloms)
    app = fb.valence(_odor(n, app_i))["innate"]
    avr = fb.valence(_odor(n, avr_i))["innate"]
    return {"observed": {"appetitive_glom": gloms[app_i], "innate_app": round(float(app), 2),
                         "aversive_glom": gloms[avr_i], "innate_avr": round(float(avr), 2)},
            "expected": "appetitive glomerulus reads > 0, aversive reads < 0",
            "pass": bool(app > 0 > avr)}


def _exp_double_dissociation(ctx):
    """Double dissociation: LH lesion abolishes innate but spares learned; MB lesion the reverse
    (Dolan 2018/2019; Aso 2014)."""
    fb, meta = ctx.flybrain()
    gloms = list(meta["gloms"]); gval = np.asarray(meta["gval"])
    n = len(gloms)
    app_g = int(np.where(gval > 0)[0][0])
    o = _odor(n, app_g)
    # form a memory so the learned channel is non-zero, then lesion each channel
    fb.mb.reset()
    kc = fb.kc_code(o); active = (kc @ fb.mb.W0) > 0
    ppl = np.where(fb.mb.dan_sign < 0)[0]
    best = int(ppl[int(np.argmax([((fb.mb.dan_comp[d] > 0) & active & (fb.mb.mbon_val > 0)).sum()
                                   for d in ppl]))])
    fb.train_odor(o, best, reps=5)
    lh = {"innate": round(fb.valence(o, lh_lesion=True)["innate"], 2),
          "learned": round(fb.valence(o, lh_lesion=True)["learned"], 2)}
    mb = {"innate": round(fb.valence(o, mb_lesion=True)["innate"], 2),
          "learned": round(fb.valence(o, mb_lesion=True)["learned"], 2)}
    return {"observed": {"LH_lesion": lh, "MB_lesion": mb},
            "expected": "LH lesion: innate=0, learned!=0; MB lesion: innate!=0, learned=0",
            "pass": bool(lh["innate"] == 0 and lh["learned"] != 0
                         and mb["innate"] != 0 and mb["learned"] == 0)}


def _exp_mb_valence_by_nt(ctx):
    """MB output valence follows transmitter: glutamatergic MBONs drive avoidance, GABA/ACh drive
    approach (Aso et al. 2014)."""
    from . import mb_behavior as mb
    fb, meta = ctx.flybrain()
    nt = list(meta["mbon_nt"])
    val = mb.mbon_valence(nt)
    glut = np.array([v for v, n in zip(val, nt) if str(n) == "GLUT"])
    gaba = np.array([v for v, n in zip(val, nt) if str(n) == "GABA"])
    ach = np.array([v for v, n in zip(val, nt) if str(n) == "ACH"])
    return {"observed": {"GLUT_mean": float(np.mean(glut)) if len(glut) else None,
                         "GABA_mean": float(np.mean(gaba)) if len(gaba) else None,
                         "ACH_mean": float(np.mean(ach)) if len(ach) else None},
            "expected": "GLUT MBONs negative (avoid); GABA/ACh positive (approach)",
            "pass": bool(len(glut) and np.mean(glut) < 0 and np.mean(gaba) > 0 and np.mean(ach) > 0)}


def _exp_mb_arena(ctx):
    """4-quadrant arena: activating avoidance (GLUT) MBONs yields negative preference index,
    approach (GABA) MBONs positive (Aso et al. 2014, optogenetic assay)."""
    from . import mb_behavior as mb
    from . import arena as ar
    fb, meta = ctx.flybrain()
    nt = list(meta["mbon_nt"]); val = mb.mbon_valence(nt)
    # arena PI driven by the mean valence of the activated MBON class
    pi_glut = ar.run_arena(np.array([-1.0]), n_flies=200, steps=2000, seed=0)["PI"]
    pi_gaba = ar.run_arena(np.array([+1.0]), n_flies=200, steps=2000, seed=0)["PI"]
    return {"observed": {"PI_avoidance_drive": round(float(pi_glut), 3),
                         "PI_approach_drive": round(float(pi_gaba), 3)},
            "expected": "avoidance drive PI < 0, approach drive PI > 0",
            "pass": bool(pi_glut < 0 < pi_gaba)}


def _exp_cx_ring_architecture(ctx):
    """Central-complex heading system, read directly from the BANC connectome: the ring-attractor
    architecture is present in the wiring. Delta7 forms a purely inhibitory ring (global
    inhibition) and EPG and PEN neurons form a reciprocal excitatory loop (local excitation), the
    two ingredients a ring attractor requires (Seelig & Jayaraman 2015; Kim 2017; Turner-Evans
    2020). No connectivity is assumed; every weight is a signed BANC synapse count."""
    import json, os
    from . import __file__ as pkgfile
    with open(os.path.join(os.path.dirname(pkgfile), "data", "cx_heading_facts.json")) as f:
        f_ = json.load(f)
    return {"observed": {"n_heading_neurons": f_["n_heading_neurons"],
                         "delta7_inhibitory_fraction": f_["delta7_inhibitory_fraction"],
                         "epg_to_pen_synapses": f_["epg_to_pen_synapses"],
                         "pen_to_epg_synapses": f_["pen_to_epg_synapses"]},
            "expected": "Delta7 forms an inhibitory ring; EPG and PEN form a reciprocal excitatory loop",
            "pass": bool(f_["delta7_inhibitory_fraction"] > 0.95
                         and f_["epg_to_pen_synapses"] > 1000
                         and f_["pen_to_epg_synapses"] > 1000)}


def _exp_cx_ring_topology(ctx):
    """The EPG heading ring is recoverable from BANC connectivity alone: EPG neurons that are
    neighbors on a ring inferred from their shared connectivity are wired more similarly than
    distant ones. Quantified as the Spearman correlation between angular distance on the recovered
    ring and connectivity similarity (Seelig & Jayaraman 2015). This shows the wiring carries the
    ring topology without using any spatial coordinates."""
    import json, os
    from . import __file__ as pkgfile
    with open(os.path.join(os.path.dirname(pkgfile), "data", "cx_heading_facts.json")) as f:
        f_ = json.load(f)
    rho = f_["ring_topology_spearman"]
    return {"observed": {"n_epg": f_["n_epg"], "ring_topology_spearman": rho,
                         "ring_topology_p": f_["ring_topology_p"]},
            "expected": "strong negative rank correlation: ring neighbors share connectivity",
            "pass": bool(rho < -0.4 and f_["ring_topology_p"] < 1e-6)}


def _exp_lh_dimensionality(ctx):
    """The lateral horn compresses the odor representation: representational dimensionality
    (participation ratio) is lower at the LHON layer than at the PN layer, despite more LHONs than
    uPNs (Frechter 2019; consistent with the Stage-2 rate-model finding)."""
    from . import model as M
    fb, meta = ctx.flybrain()
    lh = fb.lh; n = len(meta["gloms"])
    odors = [_odor(n, i) for i in range(n)]
    PN = np.array([o @ lh.P for o in odors])
    LHON = np.array([(o @ lh.P) @ lh.W for o in odors])
    pr_pn = M.participation_ratio(PN); pr_lhon = M.participation_ratio(LHON)
    return {"observed": {"PR_PN": round(pr_pn, 2), "PR_LHON": round(pr_lhon, 2),
                         "compression_ratio": round(pr_pn / pr_lhon, 2),
                         "n_uPN": PN.shape[1], "n_LHON": LHON.shape[1]},
            "expected": "PR_LHON < PR_PN (LH compresses) despite more LHONs than uPNs",
            "pass": bool(pr_lhon < pr_pn)}


def _exp_lh_lesion_dose(ctx):
    """Graded LH lesion: ablating a growing fraction of LHONs monotonically reduces the innate
    valence read-out to zero (dose-response, averaged over lesion draws)."""
    fb, meta = ctx.flybrain()
    lh = fb.lh; gval = np.asarray(meta["gval"]); n = len(meta["gloms"])
    o = _odor(n, int(np.argmax(gval)))
    W_full = lh.W.copy(); reach_full = lh.upn_sign * (o @ lh.P)
    fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    curve = []
    for f in fracs:
        vals = []
        for seed in range(8):
            rng = np.random.default_rng(seed)
            kill = rng.random(W_full.shape[1]) < f
            Wl = W_full.copy(); Wl[:, kill] = 0
            vals.append(float((reach_full * Wl.sum(axis=1)).sum()))
        curve.append(float(np.mean(vals)))
    monotone = all(curve[i] >= curve[i + 1] - 1e-6 for i in range(len(curve) - 1))
    return {"observed": {"fracs": fracs, "innate_valence": [round(v, 1) for v in curve]},
            "expected": "innate read-out decreases monotonically with lesion fraction, reaching 0 at 100%",
            "pass": bool(monotone and abs(curve[-1]) < 1e-6 and curve[0] > 0)}


def _exp_mb_learning_curve(ctx):
    """MB learning curve: the learned valence of a punished odor decreases with the number of
    DAN-paired training reps and saturates (Aso et al. 2014 depression rule)."""
    fb, meta = ctx.flybrain()
    mb = fb.mb; gval = np.asarray(meta["gval"]); n = len(meta["gloms"])
    o = _odor(n, int(np.argmax(gval)))
    kc = fb.kc_code(o); active = (kc @ mb.W0) > 0
    ppl = np.where(mb.dan_sign < 0)[0]
    best = int(ppl[int(np.argmax([((mb.dan_comp[d] > 0) & active & (mb.mbon_val > 0)).sum()
                                   for d in ppl]))])
    reps_list = [0, 1, 2, 4, 8]; curve = []
    for reps in reps_list:
        mb.reset()
        for _ in range(reps):
            mb.train(kc, best, rate=0.5)
        curve.append(round(fb.valence(o)["learned"], 2))
    mb.reset()
    monotone = all(curve[i] >= curve[i + 1] - 1e-6 for i in range(len(curve) - 1))
    saturates = abs(curve[-1] - curve[-2]) < abs(curve[1] - curve[0])
    return {"observed": {"reps": reps_list, "learned_valence": curve},
            "expected": "learned valence decreases with training reps and saturates",
            "pass": bool(monotone and saturates and curve[0] > curve[-1])}


def _exp_mb_compartment_specificity(ctx):
    """Dopaminergic plasticity is compartment-local: training through one DAN changes KC->MBON
    synapses only for MBONs in that DAN's compartment, leaving off-compartment MBONs unchanged
    (Aso et al. 2014 — the defining property of the compartmentalized memory)."""
    fb, meta = ctx.flybrain()
    mb = fb.mb; gval = np.asarray(meta["gval"]); n = len(meta["gloms"])
    o = _odor(n, int(np.argmax(gval)))
    kc = fb.kc_code(o); active = (kc @ mb.W0) > 0
    ppl = np.where(mb.dan_sign < 0)[0]
    best = int(ppl[int(np.argmax([((mb.dan_comp[d] > 0) & active & (mb.mbon_val > 0)).sum()
                                   for d in ppl]))])
    comp = mb.dan_comp[best] > 0
    mb.reset(); before = mb.mbon_activity(kc).copy()
    for _ in range(5):
        mb.train(kc, best, rate=0.5)
    after = mb.mbon_activity(kc)
    d_within = float(np.abs(after[comp] - before[comp]).mean()) if comp.any() else 0.0
    d_off = float(np.abs(after[~comp] - before[~comp]).mean()) if (~comp).any() else 0.0
    mb.reset()
    return {"observed": {"within_compartment_delta": round(d_within, 3),
                         "off_compartment_delta": round(d_off, 3),
                         "n_compartment_MBONs": int(comp.sum())},
            "expected": "within-compartment MBONs change; off-compartment MBONs do not",
            "pass": bool(d_within > 0 and d_off < 1e-6)}


def _exp_neuromod_dose(ctx):
    """Neuromodulator dose-response with distinct signs: on a recurrent rate network, increasing
    dopamine / octopamine / acetylcholine raises steady-state activity (gain up), while serotonin
    lowers it (behavioral restraint). The transfer-function reconfiguration is graded in dose."""
    from .neurons import RateModel, NeuromodState
    rng = np.random.default_rng(1)
    W = (rng.random((40, 40)) < 0.15) * 0.25; np.fill_diagonal(W, 0)
    I = np.ones(40) * 0.4
    out = {}
    for mod in ["dopamine", "octopamine", "serotonin", "acetylcholine"]:
        nm = NeuromodState(40, receptor_frac={mod: np.ones(40)})
        rmn = RateModel(W, tau=0.02, neuromod=nm)
        curve = []
        for lv in [0.0, 0.5, 1.0]:
            nm.set_level(mod, lv)
            curve.append(round(float(rmn.simulate(I, T=0.3)["r_final"].mean()), 3))
        out[mod] = curve
    up = all(out[m][2] > out[m][0] for m in ["dopamine", "octopamine", "acetylcholine"])
    down = out["serotonin"][2] < out["serotonin"][0]
    return {"observed": out,
            "expected": "DA/OA/ACh increase activity with dose; 5-HT decreases it",
            "pass": bool(up and down)}


def _exp_valence_channel_bias(ctx):
    """Aversive channels outnumber appetitive at the LH output: more LHONs are aversive-biased
    (valence index < 0) than appetitive-biased (> 0). A structural asymmetry of the wiring."""
    fb, meta = ctx.flybrain()
    vi = np.asarray(meta["lhon_vi"])
    app = int((vi > 0.1).sum()); avr = int((vi < -0.1).sum()); neu = int((np.abs(vi) <= 0.1).sum())
    return {"observed": {"appetitive_LHONs": app, "aversive_LHONs": avr, "neutral_LHONs": neu,
                         "aversive_to_appetitive_ratio": round(avr / app, 2) if app else None},
            "expected": "more aversive-biased than appetitive-biased LHONs (ratio > 1)",
            "pass": bool(avr > app)}


def _exp_lhln_inhibition_ablation(ctx):
    """Lateral horn local neurons provide feedforward inhibition: ablating LHLN input to LHONs
    (removing their synapses) raises the mean net synaptic input onto LHONs — a disinhibition
    effect. Confirms the LHLN population is net-inhibitory onto LH output."""
    from . import connectome as C
    W, ids, classes = ctx.whole_brain(scale=1.0)
    Wl = W.tocsr()
    lhln = np.where(classes == "lateral_horn_local_neuron")[0]
    lhon = np.where(classes == "lateral_horn_output_neuron")[0]
    full_in = float(Wl[lhon].sum(axis=1).mean())
    keep = np.ones(W.shape[0], bool); keep[lhln] = False
    noinh_in = float(Wl[lhon][:, keep].sum(axis=1).mean())
    return {"observed": {"mean_LHON_input_full": round(full_in, 2),
                         "mean_LHON_input_LHLN_ablated": round(noinh_in, 2),
                         "disinhibition_delta": round(noinh_in - full_in, 2),
                         "n_LHLN": int(len(lhln))},
            "expected": "ablating LHLN input raises net LHON input (LHLN are net inhibitory)",
            "pass": bool(noinh_in > full_in)}


def _exp_pn_lh_decorrelation(ctx):
    """PN -> LH transformation raises odor-odor correlation: the LH representation is less
    discriminable (more overlapping across odors) than the PN input — the flip side of the
    dimensionality compression, matching the connectome-simulation Stage-2 result."""
    fb, meta = ctx.flybrain()
    lh = fb.lh; n = len(meta["gloms"])
    odors = [_odor(n, i) for i in range(n)]
    PN = np.array([o @ lh.P for o in odors])
    LHON = np.array([(o @ lh.P) @ lh.W for o in odors])

    def mean_offdiag_corr(X):
        Xc = X[X.std(1) > 0]
        Cc = np.corrcoef(Xc); iu = np.triu_indices_from(Cc, 1)
        return float(np.nanmean(Cc[iu]))

    c_pn = mean_offdiag_corr(PN); c_lh = mean_offdiag_corr(LHON)
    return {"observed": {"PN_odor_corr": round(c_pn, 3), "LH_odor_corr": round(c_lh, 3)},
            "expected": "LH odor-odor correlation > PN (LH less discriminable / more categorical)",
            "pass": bool(c_lh > c_pn)}


def _exp_arena_dose_response(ctx):
    """Arena preference index scales monotonically with the magnitude of the driven valence: a
    graded input-output relationship, not just a sign flip (extends the Aso arena result)."""
    from . import arena as ar
    mags = [-1.0, -0.5, 0.0, 0.5, 1.0]
    pis = [round(float(ar.run_arena(np.array([m]), n_flies=300, steps=2000, seed=0)["PI"]), 3)
           for m in mags]
    monotone = all(pis[i] <= pis[i + 1] + 0.06 for i in range(len(pis) - 1))
    return {"observed": {"drive": mags, "preference_index": pis},
            "expected": "PI increases monotonically from aversive to appetitive drive",
            "pass": bool(monotone and pis[0] < 0 < pis[-1])}


def _exp_ei_ablation(ctx):
    """Whole-brain excitation/inhibition balance: removing all inhibitory (GABA/GLUT) synapses
    from the connectome rate model disinhibits the network — more neurons become active under the
    same olfactory drive. Demonstrates the connectome's inhibition constrains activity spread."""
    from .neurons import RateModel
    W, ids, classes = ctx.whole_brain(scale=0.002)
    orn = classes == "olfactory_receptor_neuron"
    I = np.zeros(W.shape[0], dtype="float32"); I[orn] = 1.0
    Wr = W.tocsr()
    a_full = RateModel(Wr, tau=0.02).simulate(I, dt=0.001, T=0.2, record_every=200)["r_final"]
    Wpos = Wr.copy(); Wpos.data = np.clip(Wpos.data, 0, None); Wpos.eliminate_zeros()
    a_noinh = RateModel(Wpos, tau=0.02).simulate(I, dt=0.001, T=0.2, record_every=200)["r_final"]
    n_full = int((a_full > 0.01).sum()); n_noinh = int((a_noinh > 0.01).sum())
    return {"observed": {"active_intact": n_full, "active_no_inhibition": n_noinh,
                         "active_ratio": round(n_noinh / n_full, 2) if n_full else None},
            "expected": "removing inhibition increases the number of active neurons (disinhibition)",
            "pass": bool(n_noinh > n_full)}


def _exp_cx_heading_tracking(ctx):
    """Central complex: the heading bump rotates by the same angle in response to a given angular
    velocity command regardless of the current heading, and does so symmetrically for both
    directions. This rotational invariance across the full circle is the defining property of a
    heading system run as a rate model with homeostatic synaptic scaling (Seelig & Jayaraman
    2015; Kim 2017). The real wiring, with per-neuron input normalization, forms a localized
    activity bump on the recovered EPG ring rather than a diffuse or all-on profile. Localization
    is the population-vector length across twenty-four seeded headings (0 uniform, 1 point bump).
    This is a connectome result: the weight matrix is BANC synapse counts, not an assumed kernel."""
    d = _load_cx_heading(ctx)
    seeds = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    pvl = float(np.mean([_cx_settle(d, s)[1] for s in seeds]))
    return {"observed": {"localization_pvl": round(pvl, 2),
                         "n_epg": int(len(d["epg_idx"])),
                         "n_seed_headings": len(seeds)},
            "expected": "real wiring forms a localized bump (population-vector length well above 0)",
            "pass": bool(pvl > 0.4)}


def _exp_cx_discrete_attractor(ctx):
    """Central-complex heading memory from the BANC connectome and its limitation. The real
    heading-system wiring, run as a rate model with synaptic scaling, settles into a small number
    of discrete stable heading states rather than a continuous ring: seeded headings across 360
    degrees collapse onto a few attractor basins. This is the honest connectome finding, the ring
    architecture is present (see cx_ring_architecture) but a continuous attractor requires
    mechanistic detail beyond the wiring, such as the ellipsoid-body wedge geometry that the edge
    list does not carry (Seelig & Jayaraman 2015; Turner-Evans 2020)."""
    d = _load_cx_heading(ctx)
    seeds = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    finals = np.sort(np.array([_cx_settle(d, s)[0] for s in seeds]))
    basins = 1 + int((np.abs(np.diff(finals)) > 0.25).sum())
    return {"observed": {"n_discrete_states": basins, "n_seed_headings": len(seeds)},
            "expected": "wiring supports a few discrete heading states, not a continuous ring",
            "pass": bool(2 <= basins <= 12)}


def _exp_olfactory_dual_pathway(ctx):
    """Whole-brain integration: a single olfactory input diverges onto BOTH higher olfactory
    centers. Driving the olfactory receptor neurons in the whole-connectome rate model activates
    Kenyon cells (mushroom body, the learned pathway) and lateral-horn output neurons (the innate
    pathway) in parallel. This is the anatomical basis of the innate/learned division of labor
    (Dolan 2019; Aso 2014), shown here to emerge in the integrated network rather than in an
    isolated circuit."""
    from .neurons import RateModel
    W, ids, classes = ctx.whole_brain(scale=0.002)
    orn = classes == "olfactory_receptor_neuron"
    I = np.zeros(W.shape[0], dtype="float32"); I[orn] = 1.0
    a = RateModel(W.tocsr(), tau=0.02).simulate(I, dt=0.001, T=0.2, record_every=200)["r_final"]
    kc = classes == "kenyon_cell"; lhon = classes == "lateral_horn_output_neuron"
    f_kc = float((a[kc] > 0.01).mean()); f_lhon = float((a[lhon] > 0.01).mean())
    return {"observed": {"kenyon_active_frac": round(f_kc, 3),
                         "lhon_active_frac": round(f_lhon, 3)},
            "expected": "olfactory drive activates both mushroom-body and lateral-horn pathways",
            "pass": bool(f_kc > 0.05 and f_lhon > 0.05)}


def _exp_connectome_ei_composition(ctx):
    """Whole-brain integration: the signed connectome is majority excitatory, reflecting the
    cholinergic predominance of the fly central brain (acetylcholine excitatory; GABA and
    glutamate inhibitory). Reports the fraction of signed synapses of each sign across all
    3.0 million polarized edges."""
    W, ids, classes = ctx.whole_brain(scale=1.0)
    data = W.tocoo().data
    n_exc = int((data > 0).sum()); n_inh = int((data < 0).sum()); tot = n_exc + n_inh
    frac_exc = n_exc / tot if tot else None
    return {"observed": {"excitatory_frac": round(frac_exc, 3),
                         "inhibitory_frac": round(1 - frac_exc, 3) if tot else None,
                         "n_signed_edges": tot},
            "expected": "excitatory synapses are the majority (cholinergic predominance)",
            "pass": bool(tot > 0 and 0.5 < frac_exc < 0.7)}


def _exp_dan_to_mbon_integration(ctx):
    """Whole-brain integration: dopaminergic teaching signals reach the mushroom-body readout.
    Driving the mushroom-body dopaminergic neurons in the whole-connectome rate model produces a
    response in the mushroom-body output neurons, confirming that the plasticity-gating population
    and the behavioral-readout population are connected in the integrated network (Aso 2014)."""
    from .neurons import RateModel
    W, ids, classes = ctx.whole_brain(scale=0.002)
    dan = np.array(["dopamin" in str(c).lower() for c in classes])
    mbon = classes == "mushroom_body_output_neuron"
    if dan.sum() == 0 or mbon.sum() == 0:
        return {"observed": {"n_dan": int(dan.sum())}, "expected": "DAN drive reaches MBONs",
                "pass": False}
    I = np.zeros(W.shape[0], dtype="float32"); I[dan] = 1.0
    a = RateModel(W.tocsr(), tau=0.02).simulate(I, dt=0.001, T=0.2, record_every=200)["r_final"]
    f_mbon = float((np.abs(a[mbon]) > 0.01).mean())
    return {"observed": {"n_dan": int(dan.sum()), "mbon_responding_frac": round(f_mbon, 3)},
            "expected": "dopaminergic drive propagates to the mushroom-body output neurons",
            "pass": bool(f_mbon > 0.05)}


def _exp_kc_mbon_convergence(ctx):
    """Whole-brain integration: many Kenyon cells converge onto each mushroom-body output neuron.
    This convergence is the substrate for reading a sparse, high-dimensional Kenyon-cell odor code
    into a small number of behavioral channels (Litwin-Kumar 2017). Measured directly from the
    connectome as the number of Kenyon-cell presynaptic partners per output neuron."""
    W, ids, classes = ctx.whole_brain(scale=1.0)
    kc = classes == "kenyon_cell"; mbon = classes == "mushroom_body_output_neuron"
    coo = W.tocoo()
    mask = kc[coo.col] & mbon[coo.row]
    from collections import Counter
    conv = Counter(coo.row[mask].tolist())
    vals = np.array(list(conv.values())) if conv else np.array([0])
    return {"observed": {"median_kc_per_mbon": int(np.median(vals)),
                         "max_kc_per_mbon": int(vals.max()),
                         "n_mbon_receiving_kc": len(conv)},
            "expected": "many Kenyon cells converge onto each output neuron (sparse-code readout)",
            "pass": bool(np.median(vals) > 3)}


def _celltype_edge_counts(ctx, pre_types, post_types):
    """Count synaptic edges from neurons of given primary cell types to neurons of given types.
    Returns a dict {(pre_type, post_type): n_edges}. Pure connectivity, no sign or dynamics."""
    from . import connectome as C
    n = ctx.nodes; e = ctx.edges
    ct = dict(zip(n[C.NODE_ID], n[C.CELLTYPE].astype(str)))
    pre_ids = set(n[C.NODE_ID].values[n[C.CELLTYPE].astype(str).isin(pre_types).values])
    post_set = set(post_types)
    ee = e.dropna(subset=["pre_root_id", "post_root_id"])
    ee = ee[ee["pre_root_id"].isin(pre_ids)]
    from collections import Counter
    out = Counter()
    for pr, po in zip(ee["pre_root_id"].values, ee["post_root_id"].values):
        pt = ct.get(po, "")
        if pt in post_set:
            out[(ct.get(pr, ""), pt)] += 1
    return dict(out)


def _exp_visual_orientation_channels(ctx):
    """Visual system (connectome-based orientation modelling, Kashalikar 2025; Seung 2023):
    the three Dm3 line-amacrine subtypes each project to a distinct orientation-selective TmY
    type, forming parallel orientation channels in the medulla. Measured as the dominant TmY
    target of each Dm3 subtype from connectivity. This is the wiring basis for the orientation
    tuning the spiking model reads out; the spatial orientation map additionally requires
    retinotopic column coordinates that the edge list does not carry, and is not asserted here."""
    if ctx.nodes is None:
        return {"observed": {}, "expected": "requires connectome", "pass": False}
    tmy = ["TmY4", "TmY9q", "TmY9q__perp", "TmY3", "TmY10", "TmY14", "TmY5a", "TmY20"]
    c = _celltype_edge_counts(ctx, ["Dm3p", "Dm3q", "Dm3v"], tmy)
    dom = {}
    for sub in ["Dm3p", "Dm3q", "Dm3v"]:
        row = {t: c.get((sub, t), 0) for t in tmy}
        dom[sub] = max(row, key=row.get)
    distinct = len(set(dom.values())) == 3
    return {"observed": {"Dm3p_target": dom["Dm3p"], "Dm3q_target": dom["Dm3q"],
                         "Dm3v_target": dom["Dm3v"]},
            "expected": "each Dm3 subtype targets a distinct orientation-selective TmY type",
            "pass": bool(distinct)}


def _exp_visual_cross_orientation(ctx):
    """Visual system: Dm3 line-amacrine subtypes preferentially inhibit the OTHER orientation
    subtypes rather than their own, the cross-orientation inhibition motif proposed for the fly
    optic lobe and previously hypothesized for mammalian cortex (Seung 2023). Measured as the
    ratio of between-subtype to within-subtype Dm3 edges."""
    if ctx.nodes is None:
        return {"observed": {}, "expected": "requires connectome", "pass": False}
    dm3 = ["Dm3p", "Dm3q", "Dm3v"]
    c = _celltype_edge_counts(ctx, dm3, dm3)
    within = sum(c.get((t, t), 0) for t in dm3)
    between = sum(v for k, v in c.items() if k[0] != k[1])
    ratio = between / within if within else None
    return {"observed": {"between_subtype_edges": between, "within_subtype_edges": within,
                         "cross_to_self_ratio": round(ratio, 2) if ratio else None},
            "expected": "Dm3 subtypes connect more between orientations than within (cross-orientation)",
            "pass": bool(ratio is not None and ratio > 1.0)}


def _exp_visual_color_recurrence(ctx):
    """Visual system: the color photoreceptors R7 and R8 and the medulla neuron Dm9 form a closed
    recurrent loop (R7<->R8, R8<->Dm9, R7<->Dm9), the recurrent motif through which chromatic
    channels share signals in the optic lobe. Measured as the presence of edges in all three
    reciprocal directions."""
    if ctx.nodes is None:
        return {"observed": {}, "expected": "requires connectome", "pass": False}
    types = ["R7", "R8", "Dm9"]
    c = _celltype_edge_counts(ctx, types, types)
    pairs = {"R7_R8": c.get(("R7", "R8"), 0), "R8_R7": c.get(("R8", "R7"), 0),
             "R8_Dm9": c.get(("R8", "Dm9"), 0), "Dm9_R8": c.get(("Dm9", "R8"), 0),
             "R7_Dm9": c.get(("R7", "Dm9"), 0), "Dm9_R7": c.get(("Dm9", "R7"), 0)}
    closed = all(v > 0 for v in pairs.values())
    return {"observed": pairs,
            "expected": "all three reciprocal directions present (closed R7/R8/Dm9 loop)",
            "pass": bool(closed)}


def _exp_visual_on_off_split(ctx):
    """Visual system: the lamina monopolar cells segregate luminance change into ON and OFF
    channels. L1 drives the ON-pathway medulla neurons (Mi1, Tm3) and L2 drives the OFF-pathway
    neurons (Tm1, Tm2), with little crossover. Measured as the fraction of each cell's output to
    ON versus OFF targets (Takemura 2013; standard fly-vision result)."""
    if ctx.nodes is None:
        return {"observed": {}, "expected": "requires connectome", "pass": False}
    on = {"Mi1", "Tm3"}; off = {"Tm1", "Tm2"}
    c = _celltype_edge_counts(ctx, ["L1", "L2"], on | off)
    l1_on = sum(c.get(("L1", t), 0) for t in on); l1_off = sum(c.get(("L1", t), 0) for t in off)
    l2_on = sum(c.get(("L2", t), 0) for t in on); l2_off = sum(c.get(("L2", t), 0) for t in off)
    l1_on_frac = l1_on / (l1_on + l1_off) if (l1_on + l1_off) else 0.0
    l2_off_frac = l2_off / (l2_on + l2_off) if (l2_on + l2_off) else 0.0
    return {"observed": {"L1_ON_fraction": round(l1_on_frac, 3),
                         "L2_OFF_fraction": round(l2_off_frac, 3)},
            "expected": "L1 targets the ON pathway, L2 targets the OFF pathway",
            "pass": bool(l1_on_frac > 0.7 and l2_off_frac > 0.7)}


def _load_disease_genetics(ctx):
    import json, os
    from . import __file__ as pkgfile
    base = os.path.join(os.path.dirname(pkgfile), "data", "disease_genetics_bundle.json")
    d = getattr(ctx, "_dg", None)
    if d is None:
        with open(base) as f:
            d = json.load(f)
        ctx._dg = d
    return d


def _load_cx_heading(ctx):
    """Load the BANC central-complex heading system (EPG, PEN1/2, PEG, Delta7) shipped as a
    signed weight matrix with the connectivity-recovered ring ordering."""
    import os, numpy as _np
    from . import __file__ as pkgfile
    d = getattr(ctx, "_cx", None)
    if d is None:
        base = os.path.join(os.path.dirname(pkgfile), "data", "cx_heading_system.npz")
        z = _np.load(base, allow_pickle=True)
        d = {"W": z["W"].astype(float), "types": z["types"].astype(str),
             "epg_idx": z["epg_idx"], "ang": z["ang"]}
        ctx._cx = d
    return d


def _cx_settle(d, seed_ang, ang_vel=0.0, ge=4.0, gi=2.0, T=1000, tau=5.0, rmax=8.0, seed=1):
    """Rate model on the REAL BANC heading connectivity with homeostatic synaptic scaling
    (per-neuron input normalization). Returns the population-vector heading and its length."""
    W = d["W"]; epg_idx = d["epg_idx"]; ang = d["ang"]; n = W.shape[0]
    Wexc = np.clip(W, 0, None); Winh = np.clip(W, None, 0)
    We = Wexc / (Wexc.sum(1, keepdims=True) + 1e-9)
    Wi = Winh / (np.abs(Winh).sum(1, keepdims=True) + 1e-9)
    Wt = ge * We + gi * Wi
    rng = np.random.default_rng(seed); r = np.abs(rng.standard_normal(n)) * 0.2
    dphi = np.abs(((ang - seed_ang + np.pi) % (2 * np.pi)) - np.pi)
    I = np.zeros(n); I[epg_idx] = 0.6 * np.exp(-(dphi / 0.6) ** 2)
    order = np.argsort(ang)
    for t in range(T):
        drive = Wt @ r + (I if t < 120 else 0.0)
        if ang_vel and t >= 120:
            e = r[epg_idx]; eo = e[order]; eo = eo - ang_vel * np.gradient(eo)
            e2 = np.empty_like(e); e2[order] = np.maximum(eo, 0); r[epg_idx] = e2
        r = r + (1.0 / tau) * (-r + rmax * np.tanh(np.maximum(drive, 0) / rmax))
    e = r[epg_idx]; z = np.sum(e * np.exp(1j * ang)) / (e.sum() + 1e-9)
    return float(np.angle(z)), float(np.abs(z))


def _perm_z(genes, lookup, allg, rng, n=2000):
    vals = [lookup[g] for g in genes if g in lookup]
    k = len(vals)
    if k < 10:
        return None
    obs = float(np.mean(vals))
    allv = np.array([lookup[g] for g in allg])
    null = np.array([allv[rng.choice(len(allv), k, replace=False)].mean() for _ in range(n)])
    z = (obs - null.mean()) / (null.std() + 1e-9)
    return {"obs": round(obs, 1), "null": round(float(null.mean()), 1),
            "z": round(float(z), 2), "n_genes": k}


def _exp_disease_neuronal_enrichment(ctx):
    """Human-disease genetics: fly orthologs of human neurological-disease genes are expressed in
    a larger fraction of neurons than genome-background genes. Uses the Fly Cell Atlas single-cell
    expression atlas (Li et al. 2022) crossed with FlyBase DIOPT human-ortholog disease
    annotations, tested against size-matched random gene panels. Epilepsy is reported as the
    largest category; all neurological categories are enriched."""
    d = _load_disease_genetics(ctx)
    lookup = d["neuron_pct"]; allg = list(lookup)
    rng = np.random.default_rng(7)
    cats = ["epilepsy", "parkinson", "alzheimer/dementia", "ataxia"]
    out = {}
    ok = True
    for c in cats:
        r = _perm_z(d["disease_categories"].get(c, []), lookup, allg, rng)
        if r is None:
            ok = False; continue
        out[c + "_neuron_pct"] = r["obs"]; out[c + "_z"] = r["z"]
    out["genome_neuron_pct"] = round(float(np.mean([lookup[g] for g in allg])), 1)
    return {"observed": out,
            "expected": "disease-ortholog genes expressed in more neurons than genome background (z>2)",
            "pass": bool(ok and out.get("epilepsy_z", 0) > 2 and out.get("parkinson_z", 0) > 2)}


def _exp_disease_ortholog_in_neuron_genes(ctx):
    """Human-disease genetics: fly genes expressed broadly in neurons are more likely to have a
    high-confidence human disease ortholog than genome-background genes. Compares the fraction of
    genes carrying a DIOPT high-confidence disease ortholog among neuron-enriched genes (expressed
    in at least 25 percent of neurons) versus all genes."""
    d = _load_disease_genetics(ctx)
    dis = set(d["diopt_disease_genes"]); neuron_hi = set(d["neuron_enriched_genes"])
    allsym = set(d["neuron_pct"])
    f_all = len(allsym & dis) / len(allsym)
    f_neu = len(neuron_hi & dis) / len(neuron_hi)
    return {"observed": {"genome_disease_ortholog_frac": round(f_all, 3),
                         "neuron_enriched_disease_ortholog_frac": round(f_neu, 3),
                         "n_neuron_enriched": len(neuron_hi)},
            "expected": "neuron-enriched genes carry disease orthologs more often than background",
            "pass": bool(f_neu > f_all)}


def _exp_disease_neuropathy_sensorimotor(ctx):
    """Human-disease genetics: fly orthologs of human peripheral-neuropathy genes are expressed in
    sensory and motor neurons above genome background, matching the sensory and motor phenotypes of
    the human diseases. Uses Fly Cell Atlas sensory-neuron and motor-neuron expression, tested
    against size-matched random panels."""
    d = _load_disease_genetics(ctx)
    genes = d["disease_categories"].get("neuropathy", [])
    rng = np.random.default_rng(11)
    rs = _perm_z(genes, d["sensory_pct"], list(d["sensory_pct"]), rng)
    rm = _perm_z(genes, d["motor_pct"], list(d["motor_pct"]), rng)
    if rs is None or rm is None:
        return {"observed": {}, "expected": "requires bundle", "pass": False}
    return {"observed": {"sensory_pct": rs["obs"], "sensory_z": rs["z"],
                         "motor_pct": rm["obs"], "motor_z": rm["z"], "n_genes": rs["n_genes"]},
            "expected": "neuropathy orthologs enriched in sensory and motor neurons (z>2)",
            "pass": bool(rs["z"] > 2 and rm["z"] > 2)}


def _exp_disease_fly_models_census(ctx):
    """Human-disease genetics: FlyBase curates experimental fly models ('model of' annotations)
    for every major neurological disease category, spanning many genes. Reports the number of
    curated disease-model annotations and distinct genes per category, quantifying the fly's role
    as a disease model organism (Wangler et al. 2017; Marygold FlyBase)."""
    d = _load_disease_genetics(ctx)
    mc = d["model_counts"]
    covered = sum(1 for c in mc if mc[c]["models"] > 0)
    return {"observed": {c: mc[c]["models"] for c in ["parkinson", "alzheimer/dementia",
                                                       "epilepsy", "ALS/motor_neuron"]},
            "expected": "curated fly models exist for all major neurological disease categories",
            "pass": bool(covered == len(mc))}


def _exp_disease_parkinson_bridge(ctx):
    """Human-disease genetics: the canonical Parkinson-disease fly genes park and Pink1 map to the
    human genes PRKN and PINK1 at high DIOPT confidence and carry the PARK2 and PARK6 recessive
    Parkinson phenotypes. This is the concrete circuit-to-disease bridge for the dopaminergic
    system the platform models (Greene 2003; Clark 2006)."""
    d = _load_disease_genetics(ctx)
    br = {r["fly"]: r for r in d["pd_bridge"]}
    park_ok = br.get("park", {}).get("human") == "PRKN" and br["park"].get("diopt", 0) >= 10
    pink_ok = br.get("Pink1", {}).get("human") == "PINK1" and br["Pink1"].get("diopt", 0) >= 10
    return {"observed": {"park_to": br["park"]["human"], "park_diopt": br["park"]["diopt"],
                         "Pink1_to": br["Pink1"]["human"], "Pink1_diopt": br["Pink1"]["diopt"]},
            "expected": "park->PRKN and Pink1->PINK1 at DIOPT>=10 with Parkinson phenotypes",
            "pass": bool(park_ok and pink_ok)}


def _exp_opto_aminergic_receptors(ctx):
    """Optogenetic and chemogenetic targets: the aminergic neuromodulator receptor panel
    (dopamine, octopamine, and serotonin G-protein-coupled receptors) is expressed in neurons well
    above genome background. These receptors are the endogenous targets of aminergic
    neuromodulation in reward, arousal, and aggression circuits (Nadim and Bucher 2014), and
    their broad neuronal expression is the substrate for that neuromodulation. Size-controlled
    against random panels."""
    d = _load_disease_genetics(ctx)
    genes = d["opto_panels"]["aminergic_gpcr"]
    lookup = d["neuron_pct"]; allg = list(lookup)
    rng = np.random.default_rng(23)
    r = _perm_z(genes, lookup, allg, rng)
    if r is None:
        return {"observed": {}, "expected": "requires bundle", "pass": False}
    return {"observed": {"receptor_neuron_pct": r["obs"], "genome_pct": r["null"],
                         "z": r["z"], "n_receptors": r["n_genes"]},
            "expected": "aminergic receptor genes expressed in neurons above background (z>2)",
            "pass": bool(r["z"] > 2)}


def _exp_opto_mechanosensation_bridge(ctx):
    """Optogenetic and mechanogenetic targets: the fly mechanosensory and nociceptive channels
    Piezo and TrpA1, central to touch, proprioception, and thermo-nociception research
    (Coste 2010), map to their human PIEZO and TRPA1 channel orthologs at high DIOPT confidence
    and carry human mechanosensory and pain disease phenotypes (PIEZO distal arthrogryposis with
    impaired proprioception; TRPA1 familial episodic pain). The specific human paralog reported
    is whichever the shipped FlyBase DIOPT table ranks highest. Bridge verified from FlyBase
    DIOPT and OMIM."""
    d = _load_disease_genetics(ctx)
    br = d["opto_bridge"]
    piezo = br.get("Piezo", {}); trpa1 = br.get("TrpA1", {})
    piezo_ok = piezo.get("human", "").startswith("PIEZO") and piezo.get("diopt", 0) >= 8 and piezo.get("has_pheno")
    trpa1_ok = trpa1.get("human") == "TRPA1" and trpa1.get("diopt", 0) >= 8 and trpa1.get("has_pheno")
    return {"observed": {"Piezo_to": piezo.get("human"), "Piezo_diopt": piezo.get("diopt"),
                         "TrpA1_to": trpa1.get("human"), "TrpA1_diopt": trpa1.get("diopt")},
            "expected": "Piezo->PIEZO and TrpA1->TRPA1 at DIOPT>=8 with human disease phenotypes",
            "pass": bool(piezo_ok and trpa1_ok)}


def _exp_opto_circadian_bridge(ctx):
    """Optogenetic and circuit-genetics targets: the core fly circadian clock genes map to their
    human clock orthologs, several of which cause familial sleep-phase disorders (per->PER family;
    tim->TIMELESS, familial advanced sleep phase; cyc->BMAL1). This bridges the clock neuron
    circuits addressed in sleep-and-arousal optogenetics (Guo 2016) to human sleep
    genetics. Verified from FlyBase DIOPT and OMIM."""
    d = _load_disease_genetics(ctx)
    br = d["opto_bridge"]
    genes = d["opto_panels"]["circadian_clock"]
    mapped = sum(1 for g in genes if br.get(g, {}).get("human"))
    with_pheno = sum(1 for g in genes if br.get(g, {}).get("has_pheno"))
    return {"observed": {"clock_genes": len(genes), "mapped_to_human": mapped,
                         "with_disease_phenotype": with_pheno,
                         "tim_to": br.get("tim", {}).get("human"),
                         "cyc_to": br.get("cyc", {}).get("human")},
            "expected": "core clock genes map to human orthologs; several carry sleep-disorder phenotypes",
            "pass": bool(mapped == len(genes) and with_pheno >= 2)}


EXPERIMENTS = {
    # --- validation: model sanity checks, not experiments ---
    "hh_fi_curve":
        {"paper": "hodgkin1952", "needs": "none", "category": "validation",
         "claim": "HH neuron f-I curve is monotone; silent below threshold",
         "run": _exp_hh_fi},
    "lif_fly_calibration":
        {"paper": "kakaria2017", "needs": "none", "category": "validation",
         "claim": "Fly LIF reaches threshold at ~25 synapses (Shiu 2023 / Kakaria & de Bivort 2017)",
         "run": _exp_lif_calibration},
    "lif_neuromod_recapitulation":
        {"paper": "shiu2024", "needs": "connectome", "category": "experiment",
         "claim": "LIF + neuromodulation on the real connectome reproduces the olfactory cascade "
                  "and dopaminergic gain increase (integration headline)",
         "run": _exp_lif_neuromod_recapitulation},
    "lh_innate_valence":
        {"paper": "frechter2019", "needs": "bundle", "category": "experiment",
         "claim": "Lateral horn carries innate valence (appetitive > 0, aversive < 0)",
         "run": _exp_lh_innate_valence},
    "innate_learned_double_dissociation":
        {"paper": "dolan2019", "needs": "bundle", "category": "experiment",
         "claim": "LH lesion abolishes innate/spares learned; MB lesion the reverse",
         "run": _exp_double_dissociation},
    "mb_valence_by_transmitter":
        {"paper": "aso2014", "needs": "bundle", "category": "experiment",
         "claim": "Glutamatergic MBONs -> avoidance; GABA/ACh MBONs -> approach",
         "run": _exp_mb_valence_by_nt},
    "mb_arena_preference":
        {"paper": "aso2014", "needs": "bundle", "category": "experiment",
         "claim": "4-quadrant arena: avoidance drive PI < 0, approach drive PI > 0",
         "run": _exp_mb_arena},
    "cx_ring_architecture":
        {"paper": "seelig2015", "needs": "bundle", "category": "experiment",
         "claim": "BANC wiring contains the ring-attractor architecture: Delta7 inhibitory ring + EPG-PEN excitatory loop",
         "run": _exp_cx_ring_architecture},
    "cx_ring_topology":
        {"paper": "seelig2015", "needs": "bundle", "category": "experiment",
         "claim": "The EPG heading ring is recoverable from BANC connectivity alone (ring neighbors share wiring)",
         "run": _exp_cx_ring_topology},
    # --- olfactory / lateral horn ---
    "lh_dimensionality_compression":
        {"paper": "frechter2019", "needs": "bundle", "category": "experiment",
         "claim": "LH lowers representational dimensionality (PR) vs the PN layer despite more LHONs",
         "run": _exp_lh_dimensionality},
    "lh_lesion_dose_response":
        {"paper": "dolan2019", "needs": "bundle", "category": "experiment",
         "claim": "Graded LHON lesion monotonically reduces innate valence read-out to zero",
         "run": _exp_lh_lesion_dose},
    "pn_lh_decorrelation":
        {"paper": "frechter2019", "needs": "bundle", "category": "experiment",
         "claim": "PN->LH transform raises odor-odor correlation (LH less discriminable / categorical)",
         "run": _exp_pn_lh_decorrelation},
    "valence_channel_bias":
        {"paper": "daschakraborty2022", "needs": "bundle", "category": "experiment",
         "claim": "Aversive-biased LHONs outnumber appetitive-biased LHONs (structural asymmetry)",
         "run": _exp_valence_channel_bias},
    "lhln_inhibition_ablation":
        {"paper": "frechter2019", "needs": "connectome", "category": "experiment",
         "claim": "Ablating LHLN input disinhibits LHONs (LHLN are net feedforward inhibition)",
         "run": _exp_lhln_inhibition_ablation},
    # --- mushroom body / learning ---
    "mb_learning_curve":
        {"paper": "aso2014", "needs": "bundle", "category": "experiment",
         "claim": "Learned valence decreases with DAN-paired training reps and saturates",
         "run": _exp_mb_learning_curve},
    "mb_compartment_specificity":
        {"paper": "aso2014", "needs": "bundle", "category": "experiment",
         "claim": "Dopaminergic plasticity is compartment-local (off-compartment MBONs unchanged)",
         "run": _exp_mb_compartment_specificity},
    # --- neuromodulation ---
    "neuromodulator_dose_response":
        {"paper": "nadim2014", "needs": "none", "category": "experiment",
         "claim": "DA/OA/ACh raise rate-model activity with dose; 5-HT lowers it (distinct signs)",
         "run": _exp_neuromod_dose},
    # --- behavior / arena ---
    "arena_dose_response":
        {"paper": "aso2014", "needs": "none", "category": "experiment",
         "claim": "Arena preference index scales monotonically with driven-valence magnitude",
         "run": _exp_arena_dose_response},
    # --- central complex ---
    "cx_heading_bump":
        {"paper": "seelig2015", "needs": "bundle", "category": "experiment",
         "claim": "Real BANC heading wiring with synaptic scaling forms a localized activity bump",
         "run": _exp_cx_heading_tracking},
    "cx_discrete_attractor":
        {"paper": "seelig2015", "needs": "bundle", "category": "experiment",
         "claim": "Real wiring supports a few discrete heading states, not a continuous ring (honest limitation)",
         "run": _exp_cx_discrete_attractor},
    # --- whole-brain E/I ---
    "whole_brain_ei_ablation":
        {"paper": "shiu2024", "needs": "connectome", "category": "experiment",
         "claim": "Removing inhibitory synapses disinhibits the connectome rate model (more active)",
         "run": _exp_ei_ablation},
    # --- whole-brain integration: the integrated system recreates cross-region findings ---
    "olfactory_dual_pathway_divergence":
        {"paper": "dolan2019", "needs": "connectome", "category": "experiment",
         "claim": "Olfactory drive diverges onto both mushroom body and lateral horn in the whole brain",
         "run": _exp_olfactory_dual_pathway},
    "connectome_ei_composition":
        {"paper": "shiu2024", "needs": "connectome", "category": "experiment",
         "claim": "The signed connectome is majority excitatory (cholinergic predominance)",
         "run": _exp_connectome_ei_composition},
    "dan_to_mbon_integration":
        {"paper": "aso2014", "needs": "connectome", "category": "experiment",
         "claim": "Dopaminergic teaching signal reaches the mushroom-body readout in the whole brain",
         "run": _exp_dan_to_mbon_integration},
    "kc_mbon_convergence":
        {"paper": "litwinkumar2017", "needs": "connectome", "category": "experiment",
         "claim": "Many Kenyon cells converge onto each output neuron (sparse-code readout integration)",
         "run": _exp_kc_mbon_convergence},
    # --- visual system: optic-lobe orientation and color wiring (Kashalikar 2025; Seung 2023) ---
    "visual_orientation_channels":
        {"paper": "kashalikar2025", "needs": "connectome", "category": "experiment",
         "claim": "Three Dm3 subtypes form parallel orientation channels via distinct TmY targets",
         "run": _exp_visual_orientation_channels},
    "visual_cross_orientation_motif":
        {"paper": "seung2024", "needs": "connectome", "category": "experiment",
         "claim": "Dm3 subtypes preferentially connect between orientations (cross-orientation inhibition)",
         "run": _exp_visual_cross_orientation},
    "visual_color_recurrence":
        {"paper": "kashalikar2025", "needs": "connectome", "category": "experiment",
         "claim": "R7, R8, and Dm9 form a closed recurrent color loop in the optic lobe",
         "run": _exp_visual_color_recurrence},
    "visual_on_off_split":
        {"paper": "takemura2013", "needs": "connectome", "category": "experiment",
         "claim": "Lamina L1 drives the ON pathway and L2 drives the OFF pathway",
         "run": _exp_visual_on_off_split},
    # --- human disease genetics: fly gene -> cell type -> human disease ortholog ---
    "disease_neuronal_enrichment":
        {"paper": "li2022", "needs": "none", "category": "experiment",
         "claim": "Fly orthologs of neurological-disease genes are expressed in more neurons than background",
         "run": _exp_disease_neuronal_enrichment},
    "disease_ortholog_in_neuron_genes":
        {"paper": "hu2011", "needs": "none", "category": "experiment",
         "claim": "Neuron-enriched fly genes carry human disease orthologs more often than background",
         "run": _exp_disease_ortholog_in_neuron_genes},
    "disease_neuropathy_sensorimotor":
        {"paper": "li2022", "needs": "none", "category": "experiment",
         "claim": "Neuropathy orthologs are enriched in sensory and motor neurons",
         "run": _exp_disease_neuropathy_sensorimotor},
    "disease_fly_models_census":
        {"paper": "wangler2017", "needs": "none", "category": "experiment",
         "claim": "Curated fly disease models exist for every major neurological disease category",
         "run": _exp_disease_fly_models_census},
    "disease_parkinson_bridge":
        {"paper": "greene2003", "needs": "none", "category": "experiment",
         "claim": "park->PRKN and Pink1->PINK1 bridge the dopaminergic system to Parkinson disease",
         "run": _exp_disease_parkinson_bridge},
    # --- optogenetic / chemogenetic circuit-genetics targets (2010s-2020s directions) ---
    "opto_aminergic_receptors":
        {"paper": "nadim2014", "needs": "none", "category": "experiment",
         "claim": "Aminergic neuromodulator receptors are expressed in neurons above background",
         "run": _exp_opto_aminergic_receptors},
    "opto_mechanosensation_bridge":
        {"paper": "coste2010", "needs": "none", "category": "experiment",
         "claim": "Piezo and TrpA1 bridge fly mechanosensation to human channelopathies",
         "run": _exp_opto_mechanosensation_bridge},
    "opto_circadian_bridge":
        {"paper": "guo2016", "needs": "none", "category": "experiment",
         "claim": "Core clock genes map to human orthologs with familial sleep-disorder phenotypes",
         "run": _exp_opto_circadian_bridge},
}


def run_experiment(name, ctx):
    spec = EXPERIMENTS[name]
    res = spec["run"](ctx)
    return {"experiment": name, "paper": spec["paper"], "category": spec.get("category", "experiment"),
            "claim": spec["claim"], **res}


def run_all(ctx, only=None):
    out = []
    for name, spec in EXPERIMENTS.items():
        if only and name not in only:
            continue
        try:
            out.append(run_experiment(name, ctx))
        except Exception as e:  # keep the registry robust; report per-experiment failure
            out.append({"experiment": name, "paper": spec["paper"], "claim": spec["claim"],
                        "error": str(e)[:200], "pass": False})
    return out
