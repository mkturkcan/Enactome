"""Multi-level neuron models for Enactome, all CPU/GPU-capable via a shared backend.

Tiers (increasing biophysical fidelity, decreasing speed):
  - rate.RateModel        — dynamical rate units with matched neuromodulation (whole-brain default)
  - spiking.LIFNetwork    — leaky integrate-and-fire, fly-calibrated (Shiu et al. 2023)
  - spiking.HHNeuron      — Hodgkin-Huxley point neuron (sample biophysical experiments)

Synapse scale and single-neuron constants for the fly are in synapse.FLY_PARAMS.
"""
from .backend import Backend, get_backend
from .rate import RateModel, NeuromodState, MODULATOR_EFFECT
from .spiking import LIFNetwork, HHNeuron
from .synapse import ExpSynapse, FLY_PARAMS, SYN_TO_THRESHOLD, as_lif_kwargs

__all__ = ["Backend", "get_backend", "RateModel", "NeuromodState", "MODULATOR_EFFECT",
           "LIFNetwork", "HHNeuron", "ExpSynapse", "FLY_PARAMS", "SYN_TO_THRESHOLD",
           "as_lif_kwargs"]
