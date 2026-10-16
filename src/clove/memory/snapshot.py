"""clove.memory.snapshot — method factory and snapshot/restore re-exports."""
from __future__ import annotations

from .append import AppendOnly, Recency
from .base import restore, snapshot
from .baselines import BM25Evidence, NamesOnly, RandomSubset, Sage0, Uniform
from .fresh import FreshState, OracleCandidates, ShuffledEvidence
from .policy import MemoryPolicy
from .prism import Prism, PrismNoProtect, PrismNoScan

REGISTRY: dict[str, type[MemoryPolicy]] = {
    cls.name: cls for cls in (NamesOnly, Sage0, AppendOnly, FreshState, Prism, Uniform, Recency,
                              RandomSubset, BM25Evidence, OracleCandidates, ShuffledEvidence,
                              PrismNoScan, PrismNoProtect)
}


def make_policy(name: str, cfg, meter, seed: int = 0) -> MemoryPolicy:
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](cfg, meter, seed)


__all__ = ["REGISTRY", "make_policy", "snapshot", "restore"]
