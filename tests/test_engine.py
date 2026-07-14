"""Regression tests: the engine must reproduce known BANC numbers and the null logic.

The connectome tests need the BANC CSVs; set ENACTOME_NODES / ENACTOME_EDGES env vars
to their paths, else those tests skip. The model/atlas tests run on synthetic data and
always execute.
"""
import os
import numpy as np
import pytest
import enactome_core as bl


def test_nt_sign():
    assert bl.connectome.NT_SIGN["ACH"] == 1
    assert bl.connectome.NT_SIGN["GABA"] == -1
    assert bl.connectome.NT_SIGN["GLUT"] == -1


def test_participation_ratio_bounds():
    # isotropic data -> PR ~ n_dims; rank-1 data -> PR ~ 1
    rng = np.random.default_rng(0)
    iso = rng.standard_normal((500, 10))
    assert bl.model.participation_ratio(iso) > 7
    v = rng.standard_normal((500, 1))
    rank1 = v @ rng.standard_normal((1, 10))
    assert bl.model.participation_ratio(rank1) < 1.5


def test_shuffle_null_on_random_wiring():
    # random wiring should NOT decode a label better than its own shuffles (z small)
    rng = np.random.default_rng(1)
    Xin = rng.random((60, 20))
    W = (rng.random((20, 40)) < 0.1) * rng.random((20, 40))
    y = (rng.random(60) < 0.5).astype(int)
    res = bl.model.shuffle_null(W, Xin, y, bl.model.rate_forward, n=50)
    assert abs(res["z"]) < 3  # no genuine structure planted


def test_enrichment_size_control():
    # A 10-gene panel of only-epilepsy genes must enrich POSITIVELY for epilepsy vs a
    # null drawn from a universe that is mostly non-epilepsy genes.
    assoc = {f"E{i}": [{"disease": "epilepsy", "score": 0.8}] for i in range(10)}
    assoc.update({f"N{i}": [{"disease": "asthma", "score": 0.8}] for i in range(40)})
    cg = {"epi_panel": [f"E{i}" for i in range(10)]}
    df = bl.atlas.enrichment(cg, assoc, n_perm=500)
    z = df["z_epilepsy/seizure"].iloc[0]
    assert z > 2  # all-epilepsy panel is enriched vs mixed-universe null
    # and a random-composition panel should NOT be enriched
    cg2 = {"mixed": [f"E{i}" for i in range(2)] + [f"N{i}" for i in range(8)]}
    z2 = bl.atlas.enrichment(cg2, assoc, n_perm=500)["z_epilepsy/seizure"].iloc[0]
    assert z2 < z


def test_mb_learning_and_ablation():
    """Aso 2014 replication on synthetic MB: aversive DAN pairing depresses approach drive,
    and ablating the compartment MBONs abolishes the learned shift."""
    import numpy as np
    from enactome_core import mb_behavior as mbb
    rng = np.random.default_rng(0)
    n_kc, n_mbon, n_dan = 200, 10, 4
    W = (rng.random((n_kc, n_mbon)) < 0.1) * rng.random((n_kc, n_mbon)) * 5
    val = np.array([+1, +1, +1, +1, +1, -1, -1, -1, 0, 0.0])  # approach/avoid/neutral
    dan_comp = np.zeros((n_dan, n_mbon)); dan_comp[0, :3] = 1   # DAN0 targets approach MBONs
    dan_sign = np.array([-1, -1, +1, +1.0])
    m = mbb.MBModel(W, val, dan_comp, dan_sign)
    odor = m.odor_code(seed=3)
    naive = m.approach_drive(odor)
    for _ in range(5):
        m.train(odor, 0, rate=0.5)
    assert m.approach_drive(odor) < naive          # learned a shift away from approach
    # ablation control
    m.reset()
    val2 = m.mbon_val.copy(); val2[dan_comp[0] > 0] = 0
    base = float(m.mbon_activity(odor) @ val2)
    for _ in range(5):
        m.train(odor, 0, rate=0.5)
    assert abs(float(m.mbon_activity(odor) @ val2) - base) < 1e-6  # memory abolished


def test_mbon_valence_from_nt():
    from enactome_core import mb_behavior as mbb
    v = mbb.mbon_valence(["GLUT", "GABA", "ACH", "DA"])
    assert list(v) == [-1.0, 1.0, 1.0, 0.0]        # Aso NT->valence rule


def test_arena_preference_signs():
    """4-quadrant arena: aversive valence in light -> negative PI (avoid);
    appetitive -> positive PI (prefer); neutral -> near zero."""
    from enactome_core import arena as ar
    pi_avoid = ar.run_arena(-1.0, n_flies=300, steps=2500, seed=1)["PI"]
    pi_appr = ar.run_arena(+1.0, n_flies=300, steps=2500, seed=1)["PI"]
    pi_neut = ar.run_arena(0.0, n_flies=300, steps=2500, seed=1)["PI"]
    assert pi_avoid < -0.1
    assert pi_appr > 0.1
    assert abs(pi_neut) < 0.1


def test_reference_ring_attractor_maintains_bump():
    """Synthetic reference RingAttractor (not connectome) holds a single localized bump.
    This tests the idealized comparison model only; the connectome heading analysis is in
    test_cx_connectome_heading."""
    import numpy as np
    from enactome_core import arena as ar
    ra = ar.RingAttractor(n=36)
    for _ in range(80):
        ra.step(0.0)
    assert ra.a.sum() > 0
    assert ra.a.max() > 3 * ra.a.mean()            # a peak exists, not flat


def test_reference_pop_ring_attractor_rotates():
    """Synthetic reference PopRingAttractor rotates its bump under angular-velocity input.
    Reference model only; not used by the arena or any connectome experiment."""
    import numpy as np
    from enactome_core import arena as ar
    cx = ar.PopRingAttractor(4, seed=0)
    for _ in range(30):
        cx.step(np.zeros(4))
    h0 = cx.heading().copy()
    for _ in range(100):
        cx.step(np.full(4, 0.1))
    assert np.all(np.abs(cx.heading() - h0) > 0.5)  # bump has rotated substantially


def test_cx_connectome_heading():
    """The connectome CX experiments run from shipped BANC wiring: ring architecture present,
    ring topology recoverable, localized bump forms, and it settles to discrete heading states."""
    import os
    from enactome_core import experiments as EX
    bundle = os.path.join(os.path.dirname(bl.__file__), "data")
    ctx = EX.ExperimentContext(bundle_dir=bundle)
    arch = EX.run_experiment("cx_ring_architecture", ctx)
    assert arch["pass"] and arch["observed"]["delta7_inhibitory_fraction"] > 0.95
    topo = EX.run_experiment("cx_ring_topology", ctx)
    assert topo["pass"] and topo["observed"]["ring_topology_spearman"] < -0.4
    bump = EX.run_experiment("cx_heading_bump", ctx)
    assert bump["pass"] and bump["observed"]["localization_pvl"] > 0.4
    disc = EX.run_experiment("cx_discrete_attractor", ctx)
    assert disc["pass"] and 2 <= disc["observed"]["n_discrete_states"] <= 12


def test_flybrain_double_dissociation():
    """The defining test of the fundamental architecture: LH lesion abolishes innate valence
    but spares learned; MB lesion the reverse. Uses the shipped connectome bundle."""
    import os
    import numpy as np
    import enactome_core as bl
    from enactome_core import olfaction as olf
    data_dir = os.path.join(os.path.dirname(bl.__file__), "data")
    if not os.path.exists(os.path.join(data_dir, "fb_meta.npz")):
        import pytest as _pt
        _pt.skip("flybrain bundle not shipped")
    brain, meta = olf.load_flybrain(data_dir)
    gval = meta["gval"]
    app_g = int(np.where(gval > 0)[0][0])
    odor = np.zeros(len(meta["gloms"])); odor[app_g] = 1.0
    # innate present with LH intact, zero with LH lesion
    brain.mb.reset()
    assert brain.valence(odor)["innate"] != 0
    assert brain.valence(odor, lh_lesion=True)["innate"] == 0
    # form a memory, then confirm MB lesion removes learned but not innate
    kc = brain.kc_code(odor); active = (kc @ brain.mb.W0) > 0
    ppl = np.where(brain.mb.dan_sign < 0)[0]
    best = int(ppl[int(np.argmax([((brain.mb.dan_comp[d] > 0) & active & (brain.mb.mbon_val > 0)).sum()
                                   for d in ppl]))])
    brain.mb.reset(); brain.train_odor(odor, best, rate=0.6, reps=8)
    v_mb_lesion = brain.valence(odor, mb_lesion=True)
    assert v_mb_lesion["learned"] == 0
    assert v_mb_lesion["innate"] != 0


def test_rate_model_settles_and_neuromodulates():
    from enactome_core import neurons as N
    rng = np.random.default_rng(0)
    W = (rng.random((15, 15)) < 0.2) * 0.3
    np.fill_diagonal(W, 0)
    I = np.ones(15) * 0.5
    rm = N.RateModel(W, tau=0.02)
    o = rm.simulate(I, dt=0.001, T=0.3)
    assert np.allclose(o["trace"][-1], o["trace"][-2], atol=1e-3)  # converged
    nm = N.NeuromodState(15, receptor_frac={"dopamine": np.ones(15)})
    rm2 = N.RateModel(W, tau=0.02, neuromod=nm)
    base = rm2.simulate(I, T=0.3)["r_final"].mean()
    nm.set_level("dopamine", 1.0)
    da = rm2.simulate(I, T=0.3)["r_final"].mean()
    assert da > base  # dopamine raises gain -> higher steady rate


def test_lif_fly_calibration():
    from enactome_core import neurons as N
    from enactome_core.neurons import synapse as syn
    # ~25 synapses reach threshold from rest (7 mV / 0.275 mV)
    assert 24 < syn.SYN_TO_THRESHOLD < 27
    kw = syn.as_lif_kwargs()
    lif = N.LIFNetwork(np.zeros((3, 3)), **kw)
    r = lif.simulate(np.array([12.0, 0, 0]), dt=0.0005, T=0.5)
    assert r["rates_hz"][0] > 20  # driven neuron fires; refractory-limited near ~45-50 Hz


def test_hh_spikes_when_driven():
    from enactome_core import neurons as N
    hh = N.HHNeuron(n=1)
    assert hh.simulate(10.0, dt=0.01, T=60.0)["rate_hz"][0] > 20
    assert hh.simulate(0.0, dt=0.01, T=60.0)["rate_hz"][0] == 0  # silent at zero current


def test_lh_perturbation_hypothesis():
    from enactome_core import olfaction as olf
    import numpy as np
    # aversive-biased LHONs (VI<0): silencing should predict increased approach
    vi = np.array([-0.5, -0.4, 0.3])
    types = np.array(["LHPV10a", "LHPV10b", "LHAV2x"])

    class _LH:  # minimal stand-in; function only reads vi/types
        pass
    r = olf.lh_perturbation_hypothesis(_LH(), vi, types, "LHPV10", mode="silence")
    assert r["n_matched"] == 2
    assert "approach" in r["predicted_behavior_shift"]
    assert "Silencing" in r["hypothesis"]


def test_experiments_bundle_only_pass():
    """The bundle/none experiments (no connectome needed) all pass from shipped data."""
    import os
    from enactome_core import experiments as EX
    bundle = os.path.join(os.path.dirname(bl.__file__), "data")
    ctx = EX.ExperimentContext(bundle_dir=bundle)
    bundle_exps = [n for n, s in EX.EXPERIMENTS.items() if s["needs"] in ("none", "bundle")]
    res = EX.run_all(ctx, only=bundle_exps)
    failed = [r["experiment"] for r in res if not r.get("pass")]
    assert not failed, f"bundle experiments failed: {failed}"


def test_lh_lesion_dose_response_monotone():
    """Ablation: graded LHON lesion monotonically reduces innate valence to 0."""
    import os
    from enactome_core import experiments as EX
    bundle = os.path.join(os.path.dirname(bl.__file__), "data")
    ctx = EX.ExperimentContext(bundle_dir=bundle)
    r = EX.run_experiment("lh_lesion_dose_response", ctx)
    vals = r["observed"]["innate_valence"]
    assert vals == sorted(vals, reverse=True)  # monotone decreasing
    assert vals[-1] == 0.0 and vals[0] > 0     # full lesion abolishes; intact positive


def test_mb_compartment_specificity_local():
    """Ablation-style: DAN plasticity changes only in-compartment MBONs."""
    import os
    from enactome_core import experiments as EX
    bundle = os.path.join(os.path.dirname(bl.__file__), "data")
    ctx = EX.ExperimentContext(bundle_dir=bundle)
    r = EX.run_experiment("mb_compartment_specificity", ctx)
    assert r["observed"]["within_compartment_delta"] > 0
    assert r["observed"]["off_compartment_delta"] == 0.0


def test_neuromodulator_signs():
    """DA/OA/ACh raise activity with dose; 5-HT lowers it."""
    from enactome_core import experiments as EX
    r = EX.run_experiment("neuromodulator_dose_response", EX.ExperimentContext())
    o = r["observed"]
    assert o["dopamine"][2] > o["dopamine"][0]
    assert o["serotonin"][2] < o["serotonin"][0]


@pytest.mark.skipif(not os.environ.get("ENACTOME_NODES"), reason="BANC data not provided")
def test_whole_brain_ei_ablation():
    """Ablation: removing inhibitory synapses increases the number of active neurons."""
    import pandas as pd
    from enactome_core import experiments as EX
    nodes = pd.read_csv(os.environ["ENACTOME_NODES"])
    edges = pd.read_csv(os.environ["ENACTOME_EDGES"])
    ctx = EX.ExperimentContext(nodes=nodes, edges=edges)
    r = EX.run_experiment("whole_brain_ei_ablation", ctx)
    assert r["observed"]["active_no_inhibition"] > r["observed"]["active_intact"]


@pytest.mark.skipif(not os.environ.get("ENACTOME_NODES"), reason="BANC data not provided")
def test_lif_neuromod_recapitulation_on_connectome():
    """HEADLINE experiment: LIF+neuromodulation on the real connectome recreates the cascade."""
    import pandas as pd
    from enactome_core import experiments as EX
    nodes = pd.read_csv(os.environ["ENACTOME_NODES"])
    edges = pd.read_csv(os.environ["ENACTOME_EDGES"])
    ctx = EX.ExperimentContext(nodes=nodes, edges=edges)
    r = EX.run_experiment("lif_neuromod_recapitulation", ctx)
    assert r["pass"], r["observed"]


def test_disease_genetics_bridges():
    """Disease-genetics bridges resolve to the expected human orthologs from shipped FlyBase data."""
    import os
    from enactome_core import experiments as EX
    bundle = os.path.join(os.path.dirname(bl.__file__), "data")
    ctx = EX.ExperimentContext(bundle_dir=bundle)
    pk = EX.run_experiment("disease_parkinson_bridge", ctx)
    assert pk["pass"] and pk["observed"]["park_to"] == "PRKN" and pk["observed"]["Pink1_to"] == "PINK1"
    mech = EX.run_experiment("opto_mechanosensation_bridge", ctx)
    assert mech["pass"] and mech["observed"]["TrpA1_to"] == "TRPA1"
    enr = EX.run_experiment("disease_neuronal_enrichment", ctx)
    assert enr["pass"] and enr["observed"]["epilepsy_z"] > 2


@pytest.mark.skipif(not os.environ.get("ENACTOME_NODES"), reason="BANC data not provided")
def test_visual_orientation_motifs():
    """Visual-system experiments recover the Dm3/TmY orientation channels and R7/R8/Dm9 loop."""
    import pandas as pd
    from enactome_core import experiments as EX
    nodes = pd.read_csv(os.environ["ENACTOME_NODES"])
    edges = pd.read_csv(os.environ["ENACTOME_EDGES"])
    ctx = EX.ExperimentContext(nodes=nodes, edges=edges)
    for name in ["visual_orientation_channels", "visual_cross_orientation_motif",
                 "visual_color_recurrence", "visual_on_off_split"]:
        assert EX.run_experiment(name, ctx)["pass"], name


@pytest.mark.skipif(not os.environ.get("ENACTOME_NODES"), reason="BANC data not provided")
def test_banc_olfactory_pathway():
    nodes, edges = bl.connectome.load_connectome(
        os.environ["ENACTOME_NODES"], os.environ["ENACTOME_EDGES"])
    tr = bl.connectome.trace_pathway(nodes, edges,
        ["olfactory_receptor_neuron", "antennal_lobe_projection_neuron", "lateral_horn_output_neuron"])
    orn_pn = tr["edges"]["olfactory_receptor_neuron->antennal_lobe_projection_neuron"]
    assert len(orn_pn) == 14121
    assert int(orn_pn.syn_count.sum()) == 190238
