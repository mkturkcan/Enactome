<div align="center">

# Enactome

**A connectome-constrained brain simulation platform — a desktop analysis app and an LLM-callable engine, from one codebase.**

📄 **[Read the paper (PDF)](paper/enactome_manuscript.pdf)** &nbsp;·&nbsp; 🎥 **[Watch the demo (YouTube)](https://youtu.be/zfk_Hc98qlk)**

</div>

## Quick start (dev)

```bash
# 1. engine
pip install -e .[server]
uvicorn server.app:app --port 8765

# 2. desktop app (separate terminal)
cd electron && npm install && npm start
```

The Electron `main.js` also auto-starts the engine, so `npm start` alone works once
`enactome-core` is installed in the active Python.

## What it is

Enactome turns a synapse-resolution connectome into fast, testable circuit models and a
circuit→gene→disease atlas. It is organism-general by design: any signed synaptic wiring diagram
can be loaded, given transmitter polarity, and evaluated through rate, integrate-and-fire, and
Hodgkin-Huxley neuron models. The current incarnation is the adult *Drosophila* brain, loaded from
the BANC / FlyWire connectome. The scientific methods are the ones developed and validated in this
project: pathway tracing, connectome-constrained rate modeling, the **shuffled-wiring null**, and
**size-controlled disease enrichment**.

**Precursors.** Enactome follows a lineage of *Drosophila* connectome platforms, in particular
FlyBrainLab (Lazar, Liu, Turkcan, Zhou, *eLife* 2021) and the Fruit Fly Brain Observatory
(Ukani et al., *bioRxiv* 2019), which established interactive access to fly connectome data and
circuit visualization. Enactome extends that lineage toward multi-level neuron simulation, a reproducible
paper-recreation experiment registry, and a circuit-to-disease atlas, on a general engine.

## Architecture — one engine, two faces

```
┌─────────────────────────┐     HTTP (localhost:8765)      ┌──────────────────────┐
│  Electron + JS UI        │ ─────────────────────────────▶│  FastAPI server      │
│  (full analysis GUI)     │                                │  (server/app.py)     │
└─────────────────────────┘                                │        │             │
┌─────────────────────────┐     same REST endpoints        │        ▼             │
│  LLM agent               │ ─────────────────────────────▶│  enactome_core       │
│  (reads /tools manifest) │                                │  (the analysis lib)  │
└─────────────────────────┘                                └──────────────────────┘
```

- **`enactome_core/`** — the engine. Pure Python (pandas/numpy/scipy/scikit-learn/networkx),
  no network, no UI. Every scientific operation lives here.
  - `connectome.py` — load, census, `trace_pathway`, signed `weight_matrix`.
  - `model.py` — `rate_forward` (divisive-norm + ReLU), `participation_ratio`, `decode_cv`,
    and `shuffle_null` (the degree/weight-preserving null that makes every claim honest).
  - `atlas.py` — size-controlled disease `enrichment` with a permutation null.
  - `mb_behavior.py` — mushroom-body learning + locomotion model (`MBModel`) that
    **replicates Aso et al. 2014**: MBON valence read from neurotransmitter identity
    (glutamatergic→avoid, GABA/cholinergic→approach), DAN-gated depression of KC→MBON
    synapses as the learning rule, and a `perturb()` **hypothesis engine** that predicts the
    behavioral effect (approach/avoidance) of activating/silencing/ablating any MBON or DAN,
    returning the genetic driver to use.
  - `arena.py` — the **canonical whole-brain demo**: a 4-quadrant optogenetic arena where the
    MB valence signal and a CX ring-attractor heading (`RingAttractor`) combine to drive
    spatial behavior, returning the preference index measured in real assays. See
    [CANONICAL_DEMO.md](CANONICAL_DEMO.md).
  - `olfaction.py` — the **olfactory pathway + lateral horn** (innate valence channel) and the
    `FlyBrain` model that combines innate (LH) + learned (MB) valence. This is the
    **canonical fly brain**: glomeruli → uPNs → {LH innate read-out ∥ KC/MB learned read-out},
    with a built-in double-dissociation test (`/flybrain/dissociation`). How every neuron and
    the dopaminergic plasticity rule are implemented is documented in
    [NEURON_MODELS.md](NEURON_MODELS.md).
- **`server/app.py`** — FastAPI wrapping the engine. Every analysis is one thin endpoint.
  `GET /tools` returns a manifest describing each endpoint as an LLM-callable tool — so an
  agent drives the *same* engine the GUI does.
- **`electron/`** — the desktop shell. `main.js` spawns the Python server as a child process
  and loads the UI; `src/` is the analysis GUI (load, census, trace, enrichment panels).

## Why this shape
The GUI and an LLM are just two clients of the same local HTTP engine. There is **no
duplicated science**: a new analysis is added once in `enactome_core`, exposed once in
`server/app.py`, and it appears in both the GUI and the `/tools` manifest.

## LLM usage
```
GET  /tools                      → list callable tools + schemas
POST /load_connectome            → {nodes_path, edges_path}
GET  /census[?group=Class]      # group optional, defaults to "Class" (the UI calls it bare)
POST /trace_pathway              → {layer_classes: [...]}
POST /enrichment                 → {circuit_genes, gene_assoc, n_perm}
POST /mb/build                   → build the MB learning+locomotion model from loaded connectome
POST /mb/perturb                 → {target_cell_type, mode} → predicted behavior + genetic handle
POST /arena                      → {target_cell_type} → 4-quadrant preference index (whole-brain demo)
POST /flybrain/valence           → {glomerulus, lh_lesion, mb_lesion, train_punishment} → innate/learned/combined valence
GET  /flybrain/dissociation      → the four canonical innate-vs-learned test cases
GET  /lh/types[?top=N]           → LHON types ranked by innate valence (both aversive & appetitive ends)
POST /lh/perturb                 → {target_type, mode} → predicted innate-behavior shift + genetic handle
GET  /neuron/models              → neuron tiers (rate/LIF/HH) + fly-calibrated params + GPU availability
POST /neuron/simulate            → {model, drive_class, prefer_gpu, T} → run rate/LIF/HH on the connectome
```

## Tests
```bash
pytest tests/                    # synthetic tests always run
ENACTOME_NODES=neurons.csv.gz ENACTOME_EDGES=connections.csv.gz pytest tests/  # + BANC regression
```
The BANC regression asserts the engine reproduces the known olfactory-pathway synapse
counts (ORN→PN 14,121 edges / 190,238 syn).

## Status
v0.4. Verification level differs by layer:
- **Engine (`enactome_core`)** — verified by the 15-test pytest suite, including a BANC regression
  that reproduces the known olfactory-pathway synapse counts, and cross-checks of every neuron tier
  (rate settles + neuromodulates; LIF fly-calibrated ~25 synapses-to-threshold; HH f–I).
- **Multi-level neuron models (`enactome_core/neurons/`)** — rate (whole-brain default, sparse),
  fly-calibrated LIF, and Hodgkin-Huxley, all CPU/GPU via a shared backend. Verified: the whole-brain
  rate model runs 158,262 neurons × 3.04M signed edges in ~0.5 s; CPU (numpy) and GPU (torch/CUDA)
  give identical results. GPU is optional (`pip install enactome-core[gpu]`); CPU is the default.
- **Server (`server/app.py`)** — 13 endpoints, verified against the real BANC connectome with a
  full-stack smoke test (in-process `TestClient`): load/census/trace/enrichment, both perturbation
  panels (`mb/perturb`, `lh/perturb`), arena, flybrain dissociation, and all three neuron tiers.
- **Electron UI (`electron/`)** — a professional multi-pane "circuit studio" (dark canvas with
  NT-colored circuit traces, left load/build/parameter rails, right live-output + perturbation rails,
  bottom drug/stimuli/experiment tray). HTML/CSS/JS are internally consistent (every element the JS
  references exists; every endpoint it calls exists on the server), but the app is **not yet launched**
  in this headless environment. `npm install && npm start` on a desktop is the first confirm.

Next: launch and smoke-test the Electron app on a desktop; bundle the Python runtime
(PyInstaller/conda-pack) for a double-click install; connectome 3D viewer; per-neuron molecular
resolution via Fly Cell Atlas scRNA.

## Provenance
The methods packaged here were developed and audited in the "Drosophila circuit modeling"
project — lateral horn valence classifier, central-complex ring attractor, and the
olfactory disease atlas. See the project reports for the science.
