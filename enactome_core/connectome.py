"""Connectome loading, census, and pathway tracing for BANC-format tables.

The engine is deliberately backend-agnostic: it operates on pandas DataFrames with
BANC column conventions (Root ID, Class, Predicted NT type, Primary Cell Type for
nodes; pre_root_id, post_root_id, syn_count for edges). Nothing here touches the
network — data are passed in as file paths or DataFrames.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

NODE_ID = "Root ID"
CLASS = "Class"
NT = "Predicted NT type"
CELLTYPE = "Primary Cell Type"

# Neurotransmitter sign convention (insect): ACh excitatory, GABA/GLUT inhibitory.
NT_SIGN = {"ACH": +1, "GABA": -1, "GLUT": -1}


def load_connectome(nodes_path: str, edges_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load node and edge tables from CSV (optionally gzipped). Strips column whitespace."""
    nodes = pd.read_csv(nodes_path, compression="infer")
    edges = pd.read_csv(edges_path, compression="infer")
    nodes.columns = [c.strip() for c in nodes.columns]
    edges.columns = [c.strip() for c in edges.columns]
    return nodes, edges


def census(nodes: pd.DataFrame, group: str = CLASS) -> pd.DataFrame:
    """Per-group neuron count + neurotransmitter composition + dominant cell types."""
    rows = []
    for g, sub in nodes.groupby(group):
        nt = sub[NT].value_counts()
        types = sub[CELLTYPE].value_counts().head(6)
        rows.append({
            "element": g,
            "n_neurons": len(sub),
            "nt_composition": {k: int(v) for k, v in nt.head(6).items()},
            "top_cell_types": {k: int(v) for k, v in types.items()},
        })
    return pd.DataFrame(rows).sort_values("n_neurons", ascending=False).reset_index(drop=True)


def trace_pathway(nodes: pd.DataFrame, edges: pd.DataFrame,
                  layer_classes: list[str]) -> dict:
    """Trace an ordered multi-layer pathway through cell-Class layers.

    Returns node ids per layer and inter-layer edge tables (layer i -> layer i+1).
    """
    layers = {c: list(nodes.loc[nodes[CLASS] == c, NODE_ID]) for c in layer_classes}
    inter = {}
    for a, b in zip(layer_classes[:-1], layer_classes[1:]):
        aset, bset = set(layers[a]), set(layers[b])
        e = edges[edges.pre_root_id.isin(aset) & edges.post_root_id.isin(bset)]
        inter[f"{a}->{b}"] = e
    return {"layers": layers, "edges": inter}


def weight_matrix(edges: pd.DataFrame, pre_ids: list, post_ids: list,
                  nt_map: dict | None = None, signed: bool = False) -> np.ndarray:
    """Synapse-count matrix (pre x post). If signed and nt_map given, multiply by NT sign."""
    pi = {p: i for i, p in enumerate(pre_ids)}
    qi = {q: i for i, q in enumerate(post_ids)}
    W = np.zeros((len(pre_ids), len(post_ids)))
    sub = edges[edges.pre_root_id.isin(set(pre_ids)) & edges.post_root_id.isin(set(post_ids))]
    for _, r in sub.iterrows():
        w = r.syn_count
        if signed and nt_map is not None:
            w *= NT_SIGN.get(nt_map.get(r.pre_root_id), 0)
        W[pi[r.pre_root_id], qi[r.post_root_id]] += w
    return W
