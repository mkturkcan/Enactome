# Enactome neuron & circuit models — what every layer computes

This is the reference for exactly how each element of the canonical fly brain is modeled,
including how **dopaminergic modulation implements memory**. Everything is rate-based (no
spikes) and connectome-constrained: the wiring is the measured BANC synapse matrix; only the
transfer functions and the plasticity rule are model choices, listed here explicitly.

Design philosophy (per the project's stated goal): neurons are *simple enough to simulate
thousands of trials in milliseconds*, but capture the *known computational property* of each
circuit. No Hodgkin-Huxley, no multicompartment cables — a deliberate choice.

---

## 1. Single-neuron model — divisive-normalized rate unit

Every neuron is a scalar firing rate `r ≥ 0`. A downstream population's rates are

    r_post = ReLU( W_pre→post · r_pre )

where `W` is the **synapse-count matrix** from the connectome (row = presynaptic, col =
postsynaptic). No per-neuron bias or time constant in the feedforward layers — the connectome
weights carry all the structure.

**Divisive normalization** (antennal lobe / PN layer, `model.divisive_norm`) models the
lateral inhibition that makes glomerular responses contrast-invariant (Olsen & Wilson):

    r_i = r_i^raw / (m + σ · Σ_j r_j^raw),   m = 0.05, σ = 0.1

**Neurotransmitter sign.** A projection is excitatory or inhibitory by the *presynaptic*
neuron's predicted transmitter (`connectome.NT_SIGN`): ACh = +1, GABA = −1, GLUT = −1
(glutamate is inhibitory at the fly LH/MB via GluCl), dopamine/serotonin = 0 in the fast
pathway (they act as modulators, see §4). Signed weight matrices multiply each presynaptic
column by its sign.

---

## 2. Lateral horn — the innate valence read-out (`olfaction.LateralHorn`)

Fixed, non-plastic. Each glomerulus carries a hardwired valence sign from the behavioural
literature (Fig. S7 of the BANC olfactory PN paper): appetitive = +1, aversive = −1,
neutral/complex/disputed = 0. A uniglomerular PN inherits its glomerulus's sign
(`upn_sign = glom_sign · P`). The innate valence of an odor is the valence-signed PN drive
pooled by how strongly each PN reaches the LH:

    innate = Σ_uPN  ( PN_drive · upn_sign · Σ_LHON W_uPN→LHON )

This is a **labeled-line** read-out: valence is carried by *which* PNs fire, and the LH is a
fixed linear pooler. Nothing here changes with experience — that is the definition of innate.

---

## 3. Mushroom body — the learned valence channel (`mb_behavior.MBModel`)

- **Kenyon cells (sparse code).** An odor is a sparse binary KC vector — top ~5% of KCs
  driven by the PN→KC projection are active (`FlyBrain.kc_code`). Sparse, high-dimensional
  coding is what makes arbitrary odors linearly separable for learning.
- **KC→MBON (plastic synapses).** `W_kc_mbon` starts at the connectome synapse counts. MBON
  activity is `ReLU(kc · W_kc_mbon)`.
- **MBON valence (Aso et al. 2014).** Each MBON's behavioral sign is read from its transmitter:
  **glutamatergic → avoidance (−1), GABAergic/cholinergic → approach (+1)**, dopaminergic/
  other → 0. The ensemble read-out (locomotion drive) is `Σ_MBON activity · valence_sign`.

---

## 4. Dopaminergic modulation = memory (the plasticity rule)

This is the mechanism the project asked to make explicit. Memory is written by
**DAN-gated depression of KC→MBON synapses** (`MBModel.train`):

    for each odor–DAN pairing:
        active_KC   = (odor's KC code > 0)
        compartment = MBONs postsynaptic to the activated DAN(s)          (from W_DAN→MBON)
        W_kc_mbon  ←  W_kc_mbon · ( 1 − η · outer(active_KC, compartment) )

That is: wherever an **active KC** and a **DAN-innervated MBON** coincide, the KC→MBON synapse
is multiplicatively depressed (learning rate η ≈ 0.5). Biological grounding:

- Dopamine is the **teaching signal**, not a fast transmitter — it does not add to MBON drive;
  it gates *plasticity* at KC→MBON synapses in its own compartment. Hence `NT_SIGN[DA] = 0`.
- **PAM cluster DANs = reward (+1), PPL1 = punishment (−1).** A punishment DAN depresses the
  KC→MBON synapses onto the *approach*-promoting MBONs of its compartment, so after training
  the odor drives less approach → net learned **avoidance**. Reward DANs do the reverse.
- The sign inversion (depression, not potentiation) reproduces Aso's observation that the
  memory written by a DAN is *opposite* in valence to activating the MBON it modulates.

**Consequences reproduced:** aversive-odor memory formation; extinction/ablation control
(silencing the compartment MBONs abolishes the learned shift); combinatorial ensemble valence.

---

## 5. Central complex — heading memory (`arena.RingAttractor`)

A ring of `n` heading units with **local excitation + global inhibition**, mirroring the BANC
compass wiring (EPG↔PEN recurrent excitation; glutamatergic Delta7 global inhibition):

    W_ij = exc · exp((cos(θ_i−θ_j) − 1)/σ²)  −  inh
    a ← a + τ(−a + ReLU(W·a)) − ω · ∂a/∂θ    (ω = angular velocity input)

The network sustains a single activity **bump** (analog heading memory) that is stable during
runs and rotates with angular velocity — the standard ring-attractor account of the fly
compass.

---

## 6. Locomotion / action selection (`arena.run_arena`)

Klinokinesis: reorientation probability rises when valence *worsens* over time,

    p_turn = clip( p0 − k · Δvalence, 0.02, 0.95 )

Runs hold the CX heading (plus small jitter); reorientations inject a large heading change.
The population's steady-state occupancy of the lit quadrants gives the **preference index**
PI ∈ [−1, +1] — the observable measured in real 4-quadrant optogenetic assays.

---

## What is connectome-derived vs. assumed

| Element | From connectome | Model assumption |
|---|---|---|
| all weight matrices | ✅ BANC synapse counts | — |
| NT sign / MBON & LH valence | ✅ predicted NT + Fig S7 | glutamate treated inhibitory / aversive |
| KC sparsity | — | top-5% (fly-realistic) |
| plasticity rule | compartments ✅ | multiplicative depression, η≈0.5 |
| ring attractor kernel | EPG/PEN/Delta7 topology ✅ | von Mises exc + uniform inh |
| locomotion controller | — | klinokinesis gain k |
