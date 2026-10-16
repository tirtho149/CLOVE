"""clove.evaluation.statistics — paired, clustered inference.

* `paired_cluster_bootstrap`: hierarchical (orders, then class-stratified specimen
  groups) bootstrap of a paired statistic; percentile CI; deterministic seed.
* `mcnemar_exact`: exact binomial test on discordant pairs.
* `sign_flip_test`: paired permutation test on cluster-level mean differences.
* `holm`, `benjamini_hochberg`: multiplicity control.
* `noninferiority`: lower CI bound vs a PRE-REGISTERED margin.
Non-significance is never reported as equivalence.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class PairedUnit:
    order_id: str
    cluster: str        # specimen / duplicate group
    stratum: str        # class
    a: float            # treatment outcome (e.g. 1 = correct)
    b: float            # control outcome
    episode: int = 0


def macro_diff(units: Sequence[PairedUnit]) -> float:
    """Per episode: macro (per-class mean, then mean over classes) of a − b; then the
    mean over episodes. Orders are pooled. This is the pre-registered primary
    statistic shape ("old-class macro accuracy averaged over post-acquisition t")."""
    by: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for u in units:
        by[u.episode][u.stratum].append(u.a - u.b)
    if not by:
        return float("nan")
    return float(np.mean([np.mean([np.mean(v) for v in d.values()]) for d in by.values()]))


def paired_cluster_bootstrap(units: Sequence[PairedUnit],
                             stat: Callable[[Sequence[PairedUnit]], float] = macro_diff,
                             n_boot: int = 10000, ci: float = 0.95, seed: int = 12345) -> dict:
    units = list(units)
    if not units:
        return {"estimate": None, "lo": None, "hi": None, "n_boot": 0}
    rng = np.random.default_rng(seed)
    orders = sorted({u.order_id for u in units})
    # order -> stratum -> cluster -> units
    tree: dict[str, dict[str, dict[str, list[PairedUnit]]]] = {}
    for u in units:
        tree.setdefault(u.order_id, {}).setdefault(u.stratum, {}).setdefault(u.cluster, []).append(u)
    est = stat(units)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample: list[PairedUnit] = []
        for o in rng.choice(orders, size=len(orders), replace=True):
            for s, clusters in tree[o].items():
                keys = list(clusters)
                for k in rng.choice(len(keys), size=len(keys), replace=True):
                    sample.extend(clusters[keys[k]])
        boots[b] = stat(sample)
    alpha = (1 - ci) / 2
    lo, hi = np.nanpercentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return {"estimate": est, "lo": float(lo), "hi": float(hi), "n_boot": n_boot, "ci": ci,
            "method": "hierarchical percentile bootstrap (orders, then class-stratified clusters)",
            "seed": seed, "n_units": len(units), "n_clusters": len({(u.order_id, u.cluster) for u in units}),
            "n_orders": len(orders), "se": float(np.nanstd(boots, ddof=1))}


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict:
    """Two-sided exact McNemar on paired binary outcomes."""
    n01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n10 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"n01": n01, "n10": n10, "p": min(1.0, 2 * p)}


def sign_flip_test(units: Sequence[PairedUnit], n_perm: int = 10000, seed: int = 0) -> dict:
    """Permutation test: within each (order, cluster) the treatment/control labels are
    exchangeable under H0, so the cluster-mean difference's sign is flipped. Exact
    enumeration when there are <= 16 clusters."""
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for u in units:
        by[(u.order_id, u.cluster)].append(u.a - u.b)
    d = np.array([np.mean(v) for v in by.values()])
    if len(d) == 0:
        return {"p": None, "n_clusters": 0}
    obs = abs(d.mean())
    if len(d) <= 16:
        signs = np.array(np.meshgrid(*[[-1, 1]] * len(d))).T.reshape(-1, len(d))
        null = np.abs((signs * d).mean(axis=1))
        p = float(np.mean(null >= obs - 1e-12))
        method = "exact"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], size=(n_perm, len(d)))
        null = np.abs((signs * d).mean(axis=1))
        p = float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1))
        method = f"monte_carlo({n_perm})"
    return {"p": p, "n_clusters": len(d), "method": method, "mean_cluster_diff": float(d.mean())}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def benjamini_hochberg(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    prev = 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        prev = min(prev, p * m / (i + 1))
        out[k] = min(1.0, prev)
    return out


def noninferiority(ci_lo: float | None, margin: float) -> dict:
    """H1: treatment − control > −margin. Claimed only if the lower CI bound clears it."""
    if ci_lo is None:
        return {"margin": margin, "noninferior": None}
    return {"margin": margin, "noninferior": bool(ci_lo > -margin), "ci_lo": ci_lo}


def paired_auroc_bootstrap(recs_a: list[dict], recs_b: list[dict], n_boot: int = 2000,
                           seed: int = 0) -> dict:
    """Paired bootstrap of AUROC(a) − AUROC(b) over shared (image, variant) items,
    resampling specimen groups."""
    from .metrics import auroc, is_known, score
    ia = {(r["image_id"], r["variant_id"]): r for r in recs_a}
    ib = {(r["image_id"], r["variant_id"]): r for r in recs_b}
    keys = sorted(set(ia) & set(ib))
    groups: dict[str, list] = defaultdict(list)
    for k in keys:
        groups[ia[k]["group_id"]].append(k)

    def diff(ks):
        pa = [score(ia[k]) for k in ks if not is_known(ia[k])]
        na = [score(ia[k]) for k in ks if is_known(ia[k])]
        pb = [score(ib[k]) for k in ks if not is_known(ib[k])]
        nb = [score(ib[k]) for k in ks if is_known(ib[k])]
        x, y = auroc(pa, na), auroc(pb, nb)
        return None if x is None or y is None else x - y

    est = diff(keys)
    if est is None:
        return {"estimate": None, "note": "NA: no eligible unknowns or knowns"}
    rng = np.random.default_rng(seed)
    g = list(groups)
    boots = []
    for _ in range(n_boot):
        ks = [k for i in rng.choice(len(g), size=len(g), replace=True) for k in groups[g[i]]]
        v = diff(ks)
        if v is not None:
            boots.append(v)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (None, None)
    n_pos = sum(1 for k in keys if not is_known(ia[k]))
    return {"estimate": est, "lo": float(lo) if lo is not None else None,
            "hi": float(hi) if hi is not None else None, "n_unknown": n_pos,
            "unstable": n_pos < 30}
