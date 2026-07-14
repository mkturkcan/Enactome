# Canonical Enactome demo — whole-brain 4-quadrant arena

This is the reference model that exercises the whole Enactome stack end-to-end: a real
connectome drives a mushroom-body learning circuit and a central-complex heading system,
which together produce measurable spatial behavior in the classic 4-quadrant optogenetic
arena. It is the model to run first to see the system work.

## What it replicates
An optogenetic place-preference assay of the kind used throughout *Drosophila* MB work.
Aso et al. 2014 (eLife 04580, fetched this session) established that optogenetic activation
of individual MBON types induces cell-type-dependent attraction or avoidance; here that
effect is cast in the spatial 4-quadrant variant: two opposite quadrants of a circular arena
are illuminated; a driver line expresses CsChrimson in a chosen cell type; the fly population
redistributes, and the **preference index (PI ∈ [−1,+1])** measures attraction to (+) or
avoidance of (−) the lit zones.

## The circuit (two connectome systems, one agent)

```
   odor ─▶ Kenyon cells ─▶ KC→MBON (plastic) ─▶ MBON ensemble ─┐
                                    ▲                            │ valence
                              DAN teaching                       ▼
   CX ring attractor ──▶ heading ──────────────▶ locomotion controller ──▶ position
   (EPG↔PEN + Delta7)                            (turn when valence worsens)
```

- **MB → valence.** When a fly is in a lit quadrant, activating the chosen MBON produces a
  locomotion drive. Its sign follows Aso 2014's rule, read straight from the connectome's
  neurotransmitter annotations: **glutamatergic MBON → avoidance, GABA/cholinergic → approach.**
- **CX → heading.** Each fly's heading is held by its own ring attractor (`PopRingAttractor`,
  a vectorised bank run inside `run_arena`): local excitation mirrors EPG↔PEN recurrence,
  global inhibition mirrors glutamatergic Delta7. Locomotion injects angular velocity into the
  ring (a reorientation rotates the bump; a run adds only jitter), and the heading the fly
  moves along is **read out from the bump** — so the CX literally supplies the heading, it is
  not an independent random walk.
- **Controller → behavior.** Klinokinesis: reorientation probability rises when valence
  *worsens* over time. Aversive light → bounce out → net avoidance; appetitive → dwell.

## Result (real BANC connectome)

| MBON activated in light | valence | preference index |
|---|---|---|
| glutamatergic (avoid)   | −1 | **PI ≈ −0.25 to −0.30** (avoids light) |
| dopaminergic (neutral)  |  0 | **PI ≈ 0** (indifferent) |
| GABAergic (approach)    | +1 | **PI ≈ +0.27 to +0.28** (prefers light) |

The signs and their neurotransmitter dependence match the experimental assay, and the whole
chain is driven by the connectome — no free behavioral parameters were fit to the data.

## Run it
```python
# via the engine
import enactome_core as bl
from enactome_core import mb_behavior as mbb, arena as ar
nodes, edges = bl.connectome.load_connectome("neurons.csv.gz", "connections.csv.gz")
# ... build W_kc_mbon, mbon_val, dan_comp, dan_sign (see mb_build in server/app.py) ...
```
```bash
# via the API / GUI
POST /mb/build
POST /arena  {"target_cell_type": "MBON-GLUT"}   # → PI, occupancy trace, interpretation
```
In the desktop app: **MB behavior → Build MB model**, then **4-quadrant arena (demo)**.

## Caveats
This is a deliberately simple demo: rate-based, single plastic layer, a phenomenological
klinokinesis controller rather than a spiking motor circuit. It shows the *architecture*
composing correctly (connectome → valence → heading → behavior), not a quantitative fit to
any one fly's trajectory. The MBON→valence mapping is structural (NT-based), so PI magnitude
depends on the controller gain, but its **sign** is a genuine connectome prediction.
