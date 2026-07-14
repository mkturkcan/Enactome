"""Enactome core: connectome-constrained circuit modeling and disease-atlas engine.

Low-level analysis library that powers both the Enactome GUI and its LLM-callable API.
All functions operate on in-memory DataFrames / arrays and never touch the network.
"""
from . import connectome, model, atlas, mb_behavior, arena, olfaction, neurons, experiments

__version__ = "0.4.0"
__all__ = ["connectome", "model", "atlas", "mb_behavior", "arena", "olfaction", "neurons",
           "experiments"]
