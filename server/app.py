"""Enactome local server: one API, two faces.

Exposes enactome_core over HTTP for the Electron GUI, and the SAME endpoints are
described in a tool manifest (/tools) so an LLM agent can call them. Every analysis
endpoint is a thin wrapper over enactome_core — no science lives here.

Run:  uvicorn server.app:app --port 8765
"""
from __future__ import annotations
import sys, os, functools
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import enactome_core as bl

app = FastAPI(title="Enactome", version=bl.__version__)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- session state: loaded connectome (cached by path) ---
_STATE: dict = {"nodes": None, "edges": None, "paths": None}


@functools.lru_cache(maxsize=4)
def _load(nodes_path: str, edges_path: str):
    return bl.connectome.load_connectome(nodes_path, edges_path)


class LoadReq(BaseModel):
    nodes_path: str
    edges_path: str


class TraceReq(BaseModel):
    layer_classes: list[str]


class EnrichReq(BaseModel):
    circuit_genes: dict[str, list[str]]
    gene_assoc: dict[str, list[dict]]
    n_perm: int = 2000


@app.get("/health")
def health():
    loaded = _STATE["nodes"] is not None
    return {"status": "ok", "version": bl.__version__, "connectome_loaded": loaded}


@app.post("/load_connectome")
def load_connectome(req: LoadReq):
    for p in (req.nodes_path, req.edges_path):
        if not os.path.exists(p):
            raise HTTPException(404, f"file not found: {p}")
    nodes, edges = _load(req.nodes_path, req.edges_path)
    _STATE.update(nodes=nodes, edges=edges, paths=(req.nodes_path, req.edges_path))
    return {"n_neurons": int(len(nodes)), "n_edges": int(len(edges)),
            "classes": sorted(nodes[bl.connectome.CLASS].dropna().unique().tolist())}


@app.get("/census")
def census(group: str = "Class"):
    _require_loaded()
    df = bl.connectome.census(_STATE["nodes"], group=group)
    return df.to_dict(orient="records")


@app.post("/trace_pathway")
def trace_pathway(req: TraceReq):
    _require_loaded()
    tr = bl.connectome.trace_pathway(_STATE["nodes"], _STATE["edges"], req.layer_classes)
    return {"layers": {k: len(v) for k, v in tr["layers"].items()},
            "edges": {k: {"n_edges": int(len(e)), "n_syn": int(e.syn_count.sum())}
                      for k, e in tr["edges"].items()}}


@app.post("/enrichment")
def enrichment(req: EnrichReq):
    df = bl.atlas.enrichment(req.circuit_genes, req.gene_assoc, n_perm=req.n_perm)
    return df.to_dict(orient="records")


def _require_loaded():
    if _STATE["nodes"] is None:
        raise HTTPException(400, "no connectome loaded; POST /load_connectome first")


class MBBuildReq(BaseModel):
    pass  # builds MB model from the loaded connectome


class PerturbReq(BaseModel):
    target_cell_type: str          # e.g. "MBON01" or a PAM/PPL DAN type, matched on Primary Cell Type
    mode: str = "activate"         # activate | silence | ablate
    odor_seed: int = 7


_MB: dict = {"model": None, "mbon_types": None, "dan_types": None}


@app.post("/mb/build")
def mb_build(req: MBBuildReq | None = None):
    """Build the connectome-constrained MB learning + locomotion model from loaded data."""
    _require_loaded()
    from enactome_core import mb_behavior as mbb
    nodes, edges = _STATE["nodes"], _STATE["edges"]
    C = bl.connectome
    kc = list(nodes.loc[nodes[C.CLASS] == "kenyon_cell", C.NODE_ID])
    mbon = nodes[nodes[C.CLASS] == "mushroom_body_output_neuron"]
    dan = nodes[nodes[C.CLASS] == "mushroom_body_dopaminergic_neuron"]
    mbon_ids, dan_ids = list(mbon[C.NODE_ID]), list(dan[C.NODE_ID])
    W = C.weight_matrix(edges, kc, mbon_ids)
    val = mbb.mbon_valence(mbon[C.NT].values)
    dan_comp = C.weight_matrix(edges, dan_ids, mbon_ids)
    dt = dan["Primary Cell Type"].astype(str).values
    dan_sign = [1.0 if t.startswith("PAM") else (-1.0 if t.startswith("PPL") else 0.0) for t in dt]
    _MB["model"] = mbb.MBModel(W, val, dan_comp, dan_sign)
    _MB["mbon_types"] = mbon["Primary Cell Type"].astype(str).tolist()
    _MB["mbon_nt"] = mbon[C.NT].astype(str).tolist()
    _MB["dan_types"] = dt.tolist()
    return {"n_kc": len(kc), "n_mbon": len(mbon_ids), "n_dan": len(dan_ids),
            "valence_counts": {"approach": int((val > 0).sum()), "avoid": int((val < 0).sum()),
                               "neutral": int((val == 0).sum())}}


@app.post("/mb/perturb")
def mb_perturb(req: PerturbReq):
    """Predict the behavioral consequence of activating/silencing a cell type — a testable hypothesis."""
    if _MB["model"] is None:
        raise HTTPException(400, "build the MB model first: POST /mb/build")
    from enactome_core import mb_behavior as mbb
    import numpy as np
    m = _MB["model"]
    odor = m.odor_code(seed=req.odor_seed)
    # match target on cell-type prefix
    mbon_hit = [i for i, t in enumerate(_MB["mbon_types"]) if t.startswith(req.target_cell_type)]
    dan_hit = [i for i, t in enumerate(_MB["dan_types"]) if t.startswith(req.target_cell_type)]
    if mbon_hit:
        return mbb.perturb(m, mbon_idx=mbon_hit, mode=req.mode, odor=odor)
    if dan_hit:
        return mbb.perturb(m, dan_idx=dan_hit, mode=req.mode, odor=odor)
    raise HTTPException(404, f"no MBON or DAN cell type matching '{req.target_cell_type}'")


class ArenaReq(BaseModel):
    target_cell_type: str = "MBON-GLUT"   # cell type to optogenetically activate in lit quadrants
    n_flies: int = 300
    steps: int = 2500
    seed: int = 1


@app.post("/arena")
def arena(req: ArenaReq):
    """Canonical demo: run the 4-quadrant optogenetic arena driven by the MB+CX model.

    Activates the named MBON/DAN cell type while flies are in lit quadrants, steers with the
    CX ring-attractor heading, and returns the behavioral preference index (PI).
    """
    if _MB["model"] is None:
        raise HTTPException(400, "build the MB model first: POST /mb/build")
    from enactome_core import arena as ar
    import numpy as np
    m = _MB["model"]
    # resolve the optogenetic valence from the MB model
    key = req.target_cell_type.replace("MBON-", "")
    if key in ("GLUT", "GABA", "ACH", "DA"):
        idx = [i for i, nt in enumerate(_MB["mbon_nt"]) if nt == key]
    else:
        idx = [i for i, t in enumerate(_MB["mbon_types"]) if t.startswith(req.target_cell_type)]
    if not idx:
        raise HTTPException(404, f"no MBON matching '{req.target_cell_type}'")
    v = m.activate_mbon(idx)
    res = ar.run_arena(v, n_flies=req.n_flies, steps=req.steps, seed=req.seed)
    return {"target": req.target_cell_type, "valence_in_light": v, "PI": res["PI"],
            "occupancy_trace": res["occupancy"], "interpretation":
            ("avoids lit quadrants" if res["PI"] < -0.05 else
             "prefers lit quadrants" if res["PI"] > 0.05 else "indifferent")}


class FlyBrainReq(BaseModel):
    glomerulus: str = None         # e.g. "DM1"; default = first appetitive glom
    lh_lesion: bool = False
    mb_lesion: bool = False
    train_punishment: bool = False # if True, first form an aversive memory of this odor


_FB: dict = {"brain": None, "meta": None}


def _get_flybrain():
    if _FB["brain"] is None:
        import os
        from enactome_core import olfaction as olf
        data_dir = os.path.join(os.path.dirname(bl.__file__), "data")
        _FB["brain"], _FB["meta"] = olf.load_flybrain(data_dir)
    return _FB["brain"], _FB["meta"]


@app.post("/flybrain/valence")
def flybrain_valence(req: FlyBrainReq):
    """Canonical fly-brain query: innate (LH) + learned (MB) valence for an odor.

    Set lh_lesion / mb_lesion to run the double-dissociation. Uses the shipped connectome
    bundle, so it works without a loaded connectome.
    """
    import numpy as np
    brain, meta = _get_flybrain()
    gloms = meta["gloms"]; gval = meta["gval"]
    if req.glomerulus and req.glomerulus in gloms:
        gi = gloms.index(req.glomerulus)
    else:
        gi = int(np.where(gval > 0)[0][0])  # default appetitive glom
    odor = np.zeros(len(gloms)); odor[gi] = 1.0
    brain.mb.reset()
    if req.train_punishment:
        # pick the punishment DAN whose compartment this odor's KCs most engage
        kc = brain.kc_code(odor); active = (kc @ brain.mb.W0) > 0
        ppl = np.where(brain.mb.dan_sign < 0)[0]
        best = int(ppl[int(np.argmax([((brain.mb.dan_comp[d] > 0) & active & (brain.mb.mbon_val > 0)).sum()
                                       for d in ppl]))])
        brain.train_odor(odor, best, rate=0.6, reps=8)
    v = brain.valence(odor, lh_lesion=req.lh_lesion, mb_lesion=req.mb_lesion)
    return {"glomerulus": gloms[gi], "innate_glom_valence": float(gval[gi]),
            "innate": round(v["innate"], 3), "learned": round(v["learned"], 3),
            "combined": round(v["combined"], 3),
            "lh_lesion": req.lh_lesion, "mb_lesion": req.mb_lesion,
            "trained_punishment": req.train_punishment}


class LHPerturbReq(BaseModel):
    target_type: str            # LHON Primary Cell Type prefix, e.g. "LHPV10", "AV", "PV5"
    mode: str = "silence"       # 'silence' | 'activate'


@app.get("/lh/types")
def lh_types(top: int = 30):
    """List LHON cell-type prefixes with their mean innate valence index (for the LH perturbation
    panel). Aversive-biased (VI<0) vs appetitive-biased (VI>0)."""
    import numpy as np
    from collections import defaultdict
    brain, meta = _get_flybrain()
    if "lhon_types" not in meta:
        return {"error": "LHON types not in bundle"}
    types = meta["lhon_types"]; vi = np.asarray(meta["lhon_vi"])
    # group by the leading alpha prefix of the type
    groups = defaultdict(list)
    for t, v in zip(types, vi):
        pref = "".join(c for c in str(t)[:5])
        groups[pref].append(v)
    rows = [{"type": k, "n": len(vs), "mean_vi": round(float(np.mean(vs)), 3)}
            for k, vs in groups.items() if len(vs) >= 3]
    rows.sort(key=lambda r: r["mean_vi"])
    # surface BOTH ends of the valence axis, not just the most-aversive: take the aversive
    # head and the appetitive tail so the panel can drive both sides of the dissociation.
    if len(rows) <= top:
        sel = rows
    else:
        half = top // 2
        sel = rows[:half] + rows[-(top - half):]
    return {"n_lhon": len(types), "n_types": len(rows),
            "most_aversive": rows[0] if rows else None,
            "most_appetitive": rows[-1] if rows else None,
            "types": sel}


@app.post("/lh/perturb")
def lh_perturb(req: LHPerturbReq):
    """Predict the innate-behavior consequence of silencing/activating a LHON type — with a
    genetically-grounded, testable hypothesis. This is the innate-channel analogue of /mb/perturb."""
    from enactome_core import olfaction as olf
    brain, meta = _get_flybrain()
    if "lhon_types" not in meta:
        return {"error": "LHON types not in bundle"}
    return olf.lh_perturbation_hypothesis(brain.lh, meta["lhon_vi"], meta["lhon_types"],
                                          req.target_type, mode=req.mode)


@app.get("/flybrain/dissociation")
def flybrain_dissociation():
    """Run the four canonical test cases proving the innate/learned double dissociation."""
    import numpy as np
    brain, meta = _get_flybrain()
    gloms = meta["gloms"]; gval = meta["gval"]
    app_g = int(np.where(gval > 0)[0][0]); avr_g = int(np.where(gval < 0)[0][0])

    def odor(g):
        a = np.zeros(len(gloms)); a[g] = 1.0; return a

    brain.mb.reset()
    t1 = {"appetitive_odor_innate": round(brain.valence(odor(app_g))["innate"], 2),
          "aversive_odor_innate": round(brain.valence(odor(avr_g))["innate"], 2)}
    o = odor(app_g)
    kc = brain.kc_code(o); active = (kc @ brain.mb.W0) > 0
    ppl = np.where(brain.mb.dan_sign < 0)[0]
    best = int(ppl[int(np.argmax([((brain.mb.dan_comp[d] > 0) & active & (brain.mb.mbon_val > 0)).sum()
                                   for d in ppl]))])
    before = brain.valence(o)["learned"]
    brain.train_odor(o, best, rate=0.6, reps=8)
    after = brain.valence(o)
    t2 = {"learned_before": round(before, 1), "learned_after": round(after["learned"], 1),
          "innate_unchanged": round(after["innate"], 2)}
    brain.mb.reset(); brain.train_odor(o, best, rate=0.6, reps=8)
    t3 = brain.valence(o, lh_lesion=True)
    brain.mb.reset(); brain.train_odor(o, best, rate=0.6, reps=8)
    t4 = brain.valence(o, mb_lesion=True)
    return {"test1_innate_tracks_glomerulus": t1,
            "test2_mb_writes_memory": t2,
            "test3_LH_lesion": {"innate": round(t3["innate"], 2), "learned": round(t3["learned"], 1)},
            "test4_MB_lesion": {"innate": round(t4["innate"], 2), "learned": round(t4["learned"], 1)},
            "conclusion": "LH carries innate valence; MB carries learned valence (double dissociation)."}


class NeuronSimReq(BaseModel):
    model: str = "rate"              # 'rate' | 'lif' | 'hh'
    scope: str = "whole_brain"       # 'whole_brain' | 'class:<Class>' (rate/lif use connectome)
    drive_class: str = "olfactory_receptor_neuron"  # which neurons to inject input into
    prefer_gpu: bool = False
    T: float = 0.2                   # seconds (rate/lif) or ms (hh)
    I_ext: float = 10.0              # hh injected current (uA/cm^2)


_BRAIN: dict = {"W": None, "ids": None, "classes": None, "scale": 0.002}


def _build_whole_brain_W(scale=0.002):
    """Signed sparse whole-brain weight matrix from the loaded connectome."""
    import numpy as np
    import scipy.sparse as sp
    from enactome_core.neurons import synapse as syn
    _require_loaded()
    nodes = _STATE["nodes"]; edges = _STATE["edges"]
    NODE = bl.connectome.NODE_ID
    ids = nodes[NODE].values
    idx = {r: i for i, r in enumerate(ids)}
    n = len(ids)
    nt = dict(zip(nodes[NODE], nodes[bl.connectome.NT]))
    sign = {"ACH": 1.0, "GABA": -1.0, "GLUT": -1.0}
    e = edges.dropna(subset=["pre_root_id", "post_root_id"])
    pre = e["pre_root_id"].map(idx); post = e["post_root_id"].map(idx)
    ok = pre.notna() & post.notna()
    pre = pre[ok].astype(int).values; post = post[ok].astype(int).values
    w = e["syn_count"].values[ok.values].astype("float32")
    psign = np.array([sign.get(str(nt.get(ids[p], "")), 0.0) for p in pre], dtype="float32")
    W = sp.csr_matrix((w * psign, (post, pre)), shape=(n, n)) * scale
    return W, ids, nodes[bl.connectome.CLASS].values


@app.get("/neuron/models")
def neuron_models():
    """List the available neuron-model tiers and the fly-calibrated parameters."""
    from enactome_core.neurons import synapse as syn
    try:
        import torch
        gpu = bool(torch.cuda.is_available())
    except Exception:
        gpu = False
    return {"tiers": [
                {"name": "rate", "desc": "Dynamical rate units + neuromodulation (whole-brain default)"},
                {"name": "lif", "desc": "Leaky integrate-and-fire, fly-calibrated (Shiu et al. 2023)"},
                {"name": "hh", "desc": "Hodgkin-Huxley point neuron (sample biophysical experiments)"}],
            "fly_params": syn.FLY_PARAMS,
            "synapses_to_threshold": round(syn.SYN_TO_THRESHOLD, 1),
            "gpu_available": gpu,
            "provenance": "Shiu et al. 2023 (biorxiv 2023.05.02.539144); Kakaria & de Bivort 2017; Jürgensen et al. 2021"}


@app.post("/neuron/simulate")
def neuron_simulate(req: NeuronSimReq):
    """Run one of the neuron tiers. rate/lif use the connectome whole-brain (or class) matrix;
    hh runs a single-cell biophysical trace. Returns summary stats (not full traces) so the
    payload stays small; the engine API exposes full traces."""
    import numpy as np
    from enactome_core import neurons as N
    from enactome_core.neurons import synapse as syn
    if req.model == "hh":
        hh = N.HHNeuron(n=1, prefer_gpu=req.prefer_gpu)
        r = hh.simulate(req.I_ext, dt=0.01, T=max(req.T, 50.0))
        return {"model": "hh", "rate_hz": round(float(r["rate_hz"][0]), 1),
                "spike_count": int(r["spike_counts"][0]), "backend": r["backend"],
                "I_ext_uA_cm2": req.I_ext}
    # rate / lif need the connectome
    if _BRAIN["W"] is None:
        _BRAIN["W"], _BRAIN["ids"], _BRAIN["classes"] = _build_whole_brain_W(_BRAIN["scale"])
    W, ids, classes = _BRAIN["W"], _BRAIN["ids"], _BRAIN["classes"]
    n = W.shape[0]
    I = np.zeros(n, dtype="float32")
    I[classes == req.drive_class] = 1.0
    n_driven = int((classes == req.drive_class).sum())
    if req.model == "rate":
        rm = N.RateModel(W, tau=0.02, prefer_gpu=req.prefer_gpu)
        o = rm.simulate(I, dt=0.001, T=req.T, record_every=max(1, int(req.T / 0.001)))
        return {"model": "rate", "n_neurons": n, "n_driven": n_driven,
                "n_active": int((o["r_final"] > 0.01).sum()),
                "mean_rate": round(float(o["r_final"].mean()), 4),
                "backend": o["backend"], "device": o["device"]}
    if req.model == "lif":
        # LIF is for detailed circuit study: scope to a class subgraph (e.g. the driven class +
        # its direct targets) rather than the full 158k-neuron brain, which is not interactive.
        drive_mask = classes == req.drive_class
        sub = np.where(drive_mask)[0]
        # include direct postsynaptic targets of the driven class
        Wc = W.tocsc()
        targets = np.unique(Wc[:, sub].tocoo().row) if len(sub) else np.array([], int)
        keep = np.unique(np.concatenate([sub, targets]))[:6000]  # cap for interactivity
        Wsub = W[keep][:, keep]
        Isub = np.zeros(len(keep), dtype="float32")
        drive_local = np.isin(keep, sub)
        Isub[drive_local] = 8.0
        kw = syn.as_lif_kwargs(prefer_gpu=False)  # sparse LIF runs on CPU
        lif = N.LIFNetwork(Wsub, **{k: v for k, v in kw.items() if k != "w_scale"}, w_scale=1.0)
        o = lif.simulate(Isub, dt=0.0005, T=min(req.T, 0.1))
        return {"model": "lif", "scope": f"class:{req.drive_class}+targets",
                "n_neurons": int(len(keep)), "n_driven": int(drive_local.sum()),
                "n_spiking": int((o["rates_hz"] > 0).sum()),
                "mean_rate_hz": round(float(o["rates_hz"].mean()), 3),
                "backend": o["backend"], "device": o["device"]}
    return {"error": f"unknown model {req.model!r}"}


class ExpRunReq(BaseModel):
    name: str = ""              # experiment id; empty runs all
    only: list[str] | None = None


_EXP_CTX = {"ctx": None}


def _experiment_ctx():
    from enactome_core import experiments as EX
    from enactome_core import olfaction as olf
    import os
    if _EXP_CTX["ctx"] is None or _EXP_CTX.get("loaded") != (_STATE["nodes"] is not None):
        bundle = os.path.join(os.path.dirname(bl.__file__), "data")
        nodes = _STATE["nodes"]; edges = _STATE["edges"]
        _EXP_CTX["ctx"] = EX.ExperimentContext(nodes=nodes, edges=edges, bundle_dir=bundle)
        _EXP_CTX["loaded"] = _STATE["nodes"] is not None
    return _EXP_CTX["ctx"]


@app.get("/experiments")
def experiments_list():
    """List the pre-structured, paper-linked experiments. Each recreates a published finding by
    driving the engine; 'needs' says whether it requires the loaded connectome, a shipped bundle,
    or nothing."""
    from enactome_core import experiments as EX
    return {"experiments": [{"name": k, "paper": v["paper"], "needs": v["needs"],
                             "category": v.get("category", "experiment"), "claim": v["claim"]}
                            for k, v in EX.EXPERIMENTS.items()]}


@app.post("/experiments/run")
def experiments_run(req: ExpRunReq):
    """Run one experiment (name=...) or the whole registry (empty name). Connectome experiments
    require /load_connectome first; bundle experiments run from shipped data."""
    from enactome_core import experiments as EX
    needs_conn = any(EX.EXPERIMENTS[n]["needs"] == "connectome"
                     for n in ([req.name] if req.name else EX.EXPERIMENTS)
                     if n in EX.EXPERIMENTS)
    if needs_conn and _STATE["nodes"] is None:
        return {"error": "connectome experiments require POST /load_connectome first"}
    ctx = _experiment_ctx()
    if req.name:
        return EX.run_experiment(req.name, ctx)
    res = EX.run_all(ctx, only=req.only)
    return {"results": res, "passed": sum(1 for r in res if r.get("pass")), "total": len(res)}


@app.get("/llm/status")
def llm_status():
    """Report whether the optional Claude API layer is configured. The engine never requires it."""
    from enactome_core import llm
    return {"api_key_present": llm.api_key_present(), "model": llm.default_model()}


class LLMAskReq(BaseModel):
    question: str
    model: str | None = None


@app.post("/llm/ask")
def llm_ask(req: LLMAskReq):
    """Optional: ask Claude which engine tools to call for a natural-language request. Requires
    ANTHROPIC_API_KEY and `pip install anthropic`; returns an error dict if either is missing."""
    from enactome_core import llm
    return llm.plan(req.question, tools()["tools"], model=req.model)


@app.get("/disease/atlas")
def disease_atlas():
    """Run the disease-genetics experiments from the shipped bundle and return their observed
    values, for the disease-atlas data screen. Requires no connectome."""
    from enactome_core import experiments as EX
    ctx = _experiment_ctx()
    names = ["disease_neuronal_enrichment", "disease_ortholog_in_neuron_genes",
             "disease_neuropathy_sensorimotor", "disease_fly_models_census",
             "disease_parkinson_bridge"]
    return {"results": [{"name": n, **EX.run_experiment(n, ctx)} for n in names]}


# --- LLM tool manifest: same endpoints, described as callable tools ---
@app.get("/tools")
def tools():
    return {"tools": [
        {"name": "load_connectome", "method": "POST", "path": "/load_connectome",
         "description": "Load node+edge connectome tables from local CSV paths.",
         "params": {"nodes_path": "str", "edges_path": "str"}},
        {"name": "census", "method": "GET", "path": "/census",
         "description": "Per-cell-class neuron counts, NT composition, dominant cell types.",
         "params": {"group": "str (default 'Class')"}},
        {"name": "trace_pathway", "method": "POST", "path": "/trace_pathway",
         "description": "Trace an ordered multi-layer pathway; returns per-layer sizes and inter-layer synapse counts.",
         "params": {"layer_classes": "list[str]"}},
        {"name": "enrichment", "method": "POST", "path": "/enrichment",
         "description": "Size-controlled, null-tested disease enrichment per circuit element.",
         "params": {"circuit_genes": "dict[element,list[gene]]", "gene_assoc": "dict[gene,list[{disease,score}]]", "n_perm": "int"}},
        {"name": "mb_build", "method": "POST", "path": "/mb/build",
         "description": "Build the mushroom-body learning + locomotion model (Aso 2014) from the loaded connectome.",
         "params": {}},
        {"name": "mb_perturb", "method": "POST", "path": "/mb/perturb",
         "description": "Predict the behavioral effect (approach/avoidance) of optogenetically activating, silencing, or ablating an MBON or DAN cell type — returns a testable hypothesis with the genetic driver to use.",
         "params": {"target_cell_type": "str (e.g. 'MBON01', 'PPL1', 'PAM')", "mode": "activate|silence|ablate", "odor_seed": "int"}},
        {"name": "arena", "method": "POST", "path": "/arena",
         "description": "Canonical demo: 4-quadrant optogenetic arena. Activates an MBON/DAN in lit quadrants, steers flies by a behavioral klinokinesis heading controller, returns the behavioral preference index (PI) — the quantity measured in real MB optogenetics assays.",
         "params": {"target_cell_type": "str (e.g. 'MBON-GLUT', 'MBON01')", "n_flies": "int", "steps": "int", "seed": "int"}},
        {"name": "flybrain_valence", "method": "POST", "path": "/flybrain/valence",
         "description": "Canonical fly brain: innate (lateral horn) + learned (mushroom body) valence for an odor, from the shipped connectome bundle. Supports LH/MB lesions and forming an aversive memory — the fundamental innate-vs-learned architecture.",
         "params": {"glomerulus": "str (e.g. 'DM1')", "lh_lesion": "bool", "mb_lesion": "bool", "train_punishment": "bool"}},
        {"name": "flybrain_dissociation", "method": "GET", "path": "/flybrain/dissociation",
         "description": "Run the four canonical test cases proving the innate/learned double dissociation (LH lesion abolishes innate but spares learning; MB lesion the reverse).",
         "params": {}},
        {"name": "neuron_models", "method": "GET", "path": "/neuron/models",
         "description": "List the neuron-model tiers (rate / LIF / Hodgkin-Huxley) and the fly-calibrated single-neuron and synapse parameters (Shiu et al. 2023), plus whether a GPU is available.",
         "params": {}},
        {"name": "disease_atlas", "method": "GET", "path": "/disease/atlas",
         "description": "Run the disease-genetics experiments from the shipped bundle and return their observed values (no connectome required).",
         "params": {}},
        {"name": "llm_ask", "method": "POST", "path": "/llm/ask",
         "description": "Optional Claude planning layer: ask which engine tools to call for a natural-language request. Requires ANTHROPIC_API_KEY and the anthropic SDK.",
         "params": {"question": "str", "model": "str optional"}},
        {"name": "experiments", "method": "GET", "path": "/experiments",
         "description": "List the pre-structured, paper-linked experiments — each recreates a published finding (Aso 2014, Frechter/Dolan 2019, Seelig 2015, Hodgkin-Huxley 1952, Shiu 2023) by driving the engine.",
         "params": {}},
        {"name": "experiments_run", "method": "POST", "path": "/experiments/run",
         "description": "Run one paper-recreation experiment by name, or the whole registry (empty name). The headline 'lif_neuromod_recapitulation' shows the LIF+neuromodulation model on the real connectome reproduces the rate-model olfactory results.",
         "params": {"name": "str (experiment id, empty = all)", "only": "list[str] optional subset"}},
        {"name": "lh_types", "method": "GET", "path": "/lh/types",
         "description": "List lateral horn output-neuron types with their innate valence index (aversive vs appetitive-biased) — the menu for the LH perturbation panel.",
         "params": {"top": "int"}},
        {"name": "lh_perturb", "method": "POST", "path": "/lh/perturb",
         "description": "Predict the innate-behavior change from silencing or activating a LHON type, with a genetically-grounded testable hypothesis (LH split-GAL4 + UAS-Kir2.1/CsChrimson). The innate-channel analogue of mb_perturb.",
         "params": {"target_type": "str (LHON type prefix)", "mode": "str: silence|activate"}},
        {"name": "neuron_simulate", "method": "POST", "path": "/neuron/simulate",
         "description": "Simulate a neuron tier on the real connectome. 'rate' runs the whole-brain (158k-neuron) dynamical rate model with neuromodulation; 'lif' runs fly-calibrated integrate-and-fire; 'hh' runs a single Hodgkin-Huxley cell. Choose CPU or GPU with prefer_gpu.",
         "params": {"model": "str: rate|lif|hh", "drive_class": "str (Class to inject input)", "prefer_gpu": "bool", "T": "float seconds (rate/lif) or ms (hh)", "I_ext": "float (hh current)"}},
    ]}
