# Enactome experiment registry

The registry contains **35 experiments** and **2 validation checks**. Every experiment recreates a finding from the published literature using the connectome, the shipped data bundles, or both. `needs` indicates the data required: `none` (self-contained), `bundle` (shipped reference data), or `connectome` (the BANC node and edge tables).

Run all: `enactome run-experiments`. Run one: `enactome run <name>`.

## Validation

| Experiment | Needs | Reference | Result |
|---|---|---|---|
| `hh_fi_curve` | none | hodgkin1952 | rate_at_0=0.0; rate_at_10=66.66666666666667 |
| `lif_fly_calibration` | none | kakaria2017 | synapses_to_threshold=25.5; driven_rate_hz=50.0 |

## Whole-brain integration

| Experiment | Needs | Reference | Result |
|---|---|---|---|
| `connectome_ei_composition` | connectome | shiu2024 | excitatory_frac=0.574; inhibitory_frac=0.426; n_signed_edges=2725094 |
| `dan_to_mbon_integration` | connectome | aso2014 | n_dan=301; mbon_responding_frac=0.279 |
| `kc_mbon_convergence` | connectome | litwinkumar2017 | median_kc_per_mbon=13; max_kc_per_mbon=440; n_mbon_receiving_kc=75 |
| `lif_neuromod_recapitulation` | connectome | shiu2024 | ORN_hz=16.44; PN_hz=81.78; LHON_hz=41.15; LHON_hz_high_gain=48.67 |
| `olfactory_dual_pathway_divergence` | connectome | dolan2019 | kenyon_active_frac=0.904; lhon_active_frac=0.537 |
| `whole_brain_ei_ablation` | connectome | shiu2024 | active_intact=10930; active_no_inhibition=15886; active_ratio=1.45 |

## Lateral horn

| Experiment | Needs | Reference | Result |
|---|---|---|---|
| `lh_dimensionality_compression` | bundle | frechter2019 | PR_PN=2.74; PR_LHON=1.71; compression_ratio=1.6; n_uPN=451 |
| `lh_innate_valence` | bundle | frechter2019 | appetitive_glom=DA3; innate_app=81.0; aversive_glom=D; innate_avr=-410.0 |
| `lh_lesion_dose_response` | bundle | dolan2019 | fracs=[0.0, 0.25, 0.5, 0.75, 1.0]; innate_valence=[81.0, 62.1, 38.1, 17.5, 0.0] |
| `lhln_inhibition_ablation` | connectome | frechter2019 | mean_LHON_input_full=6.66; mean_LHON_input_LHLN_ablated=11.8; disinhibition_delta=5.14; n_LHLN=577 |
| `pn_lh_decorrelation` | bundle | frechter2019 | PN_odor_corr=-0.015; LH_odor_corr=0.137 |
| `valence_channel_bias` | bundle | daschakraborty2022 | appetitive_LHONs=127; aversive_LHONs=285; neutral_LHONs=667; aversive_to_appetitive_ratio=2.24 |

## Lateral horn vs mushroom body

| Experiment | Needs | Reference | Result |
|---|---|---|---|
| `innate_learned_double_dissociation` | bundle | dolan2019 | LH_lesion={'innate': 0.0, 'learned': 13.59}; MB_lesion={'innate': 81.0, 'learned': 0.0} |

## Mushroom body

| Experiment | Needs | Reference | Result |
|---|---|---|---|
| `arena_dose_response` | none | aso2014 | drive=[-1.0, -0.5, 0.0, 0.5, 1.0]; preference_index=[-0.262, -0.259, 0.013, 0.424, 0.424] |
| `mb_arena_preference` | bundle | aso2014 | PI_avoidance_drive=-0.25; PI_approach_drive=0.392 |
| `mb_compartment_specificity` | bundle | aso2014 | within_compartment_delta=6.135; off_compartment_delta=0.0; n_compartment_MBONs=3 |
| `mb_learning_curve` | bundle | aso2014 | reps=[0, 1, 2, 4, 8]; learned_valence=[32.0, 22.5, 17.75, 14.19, 13.07] |
| `mb_valence_by_transmitter` | bundle | aso2014 | GLUT_mean=-1.0; GABA_mean=1.0; ACH_mean=1.0 |

## Neuromodulation

| Experiment | Needs | Reference | Result |
|---|---|---|---|
| `neuromodulator_dose_response` | none | nadim2014 | dopamine=[0.904, 0.96, 0.982]; octopamine=[0.904, 0.959, 0.981]; serotonin=[0.904, 0.844, 0.74]; acetylcholine=[0.904, 0.939, 0.96] |

## Central complex

| Experiment | Needs | Reference | Result |
|---|---|---|---|
| `cx_discrete_attractor` | bundle | seelig2015 | n_discrete_states=4; n_seed_headings=24 |
| `cx_heading_bump` | bundle | seelig2015 | localization_pvl=0.63; n_epg=50; n_seed_headings=24 |
| `cx_ring_architecture` | bundle | seelig2015 | n_heading_neurons=151; delta7_inhibitory_fraction=1.0; epg_to_pen_synapses=7998; pen_to_epg_synapses=6151 |
| `cx_ring_topology` | bundle | seelig2015 | n_epg=50; ring_topology_spearman=-0.818; ring_topology_p=2.83398390132171e-296 |

## Visual system

| Experiment | Needs | Reference | Result |
|---|---|---|---|
| `visual_color_recurrence` | connectome | kashalikar2025 | R7_R8=504; R8_R7=828; R8_Dm9=57; Dm9_R8=35 |
| `visual_cross_orientation_motif` | connectome | seung2024 | between_subtype_edges=781; within_subtype_edges=85; cross_to_self_ratio=9.19 |
| `visual_on_off_split` | connectome | takemura2013 | L1_ON_fraction=0.994; L2_OFF_fraction=0.929 |
| `visual_orientation_channels` | connectome | kashalikar2025 | Dm3p_target=TmY9q; Dm3q_target=TmY9q__perp; Dm3v_target=TmY4 |

## Human disease genetics

| Experiment | Needs | Reference | Result |
|---|---|---|---|
| `disease_fly_models_census` | none | wangler2017 | parkinson=1489; alzheimer/dementia=1867; epilepsy=687; ALS/motor_neuron=2216 |
| `disease_neuronal_enrichment` | none | li2022 | epilepsy_neuron_pct=11.3; epilepsy_z=13.24; parkinson_neuron_pct=11.2; parkinson_z=4.0 |
| `disease_neuropathy_sensorimotor` | none | li2022 | sensory_pct=10.9; sensory_z=7.69; motor_pct=11.9; motor_z=6.68 |
| `disease_ortholog_in_neuron_genes` | none | hu2011 | genome_disease_ortholog_frac=0.293; neuron_enriched_disease_ortholog_frac=0.496; n_neuron_enriched=1083 |
| `disease_parkinson_bridge` | none | greene2003 | park_to=PRKN; park_diopt=14; Pink1_to=PINK1; Pink1_diopt=11 |

## Optogenetic / circuit-genetics targets

| Experiment | Needs | Reference | Result |
|---|---|---|---|
| `opto_aminergic_receptors` | none | nadim2014 | receptor_neuron_pct=20.1; genome_pct=5.4; z=4.25; n_receptors=11 |
| `opto_circadian_bridge` | none | guo2016 | clock_genes=5; mapped_to_human=5; with_disease_phenotype=5; tim_to=TIMELESS |
| `opto_mechanosensation_bridge` | none | coste2010 | Piezo_to=PIEZO1; Piezo_diopt=12; TrpA1_to=TRPA1; TrpA1_diopt=14 |
