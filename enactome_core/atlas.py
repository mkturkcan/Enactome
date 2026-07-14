"""Circuit -> gene -> human ortholog -> disease enrichment (size-controlled).

The disease-atlas bridge. Ortholog mapping and disease-association fetching are done
externally (via MCP connectors) and passed in as tables; this module does the
size-controlled enrichment with the permutation null — the statistically load-bearing
step the earlier raw-burden metric got wrong.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Neuro-disease keyword buckets (disease name -> category).
DISEASE_BUCKETS = {
    "epilepsy/seizure": ["epilep", "seizure", "encephalopathy", "convuls"],
    "parkinsonism/dystonia": ["parkinson", "dystonia", "dopa-responsive"],
    "alzheimer/dementia": ["alzheimer", "dementia", "neurodegener"],
    "autism/neurodevelopmental": ["autism", "neurodevelopmental", "intellectual disab", "fragile"],
    "schizophrenia/psychosis": ["schizophrenia", "psychosis", "psychotic"],
    "depression/anxiety": ["depress", "anxiety", "bipolar", "mood"],
    "ADHD": ["attention deficit", "hyperactiv"],
    "addiction": ["addiction", "substance", "nicotine depend", "cocaine", "alcohol"],
}


def gene_bucket_vector(assoc_rows: list[dict], buckets: dict = DISEASE_BUCKETS) -> np.ndarray:
    """Max association score per disease bucket for one gene's association rows."""
    v = np.zeros(len(buckets))
    for row in assoc_rows:
        name = row["disease"].lower()
        for j, kws in enumerate(buckets.values()):
            if any(k in name for k in kws):
                v[j] = max(v[j], row["score"])
    return v


def enrichment(circuit_genes: dict[str, list[str]], gene_assoc: dict[str, list[dict]],
               buckets: dict = DISEASE_BUCKETS, n_perm: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Size-controlled disease enrichment per circuit element.

    circuit_genes: element -> list of human gene symbols in its machinery.
    gene_assoc: human gene -> list of {disease, score} rows.
    Returns per-(element, bucket) enrichment z-score vs same-size random gene panels.
    """
    bnames = list(buckets)
    all_genes = sorted(gene_assoc)
    gidx = {g: i for i, g in enumerate(all_genes)}
    G = np.array([gene_bucket_vector(gene_assoc[g], buckets) for g in all_genes])
    rng = np.random.default_rng(seed)
    rows = []
    for elem, genes in circuit_genes.items():
        idx = [gidx[g] for g in genes if g in gidx]
        if not idx:
            continue
        k = len(idx)
        obs = G[idx].mean(0)
        null = np.array([G[rng.choice(len(all_genes), k, replace=False)].mean(0)
                         for _ in range(n_perm)])
        z = (obs - null.mean(0)) / (null.std(0) + 1e-9)
        row = {"circuit_element": elem, "n_genes": k}
        for j, b in enumerate(bnames):
            row[f"z_{b}"] = round(float(z[j]), 2)
        rows.append(row)
    return pd.DataFrame(rows)
