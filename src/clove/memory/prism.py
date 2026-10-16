"""clove.memory.prism — Provenance-aware Retrieval and Interference-Sensitive Memory.

Deterministic, non-learned. Three mechanisms on top of a persistent memory:

1. Interference-triggered protection (update time). When classes Δ_t are admitted,
   `scan_interference` scores each new class against admitted ones; old classes whose
   score exceeds `interference_threshold` (top `refresh_top_k` per new class) get
   their card recompiled against the current inventory, so source-cited contrasts
   with the newcomer can enter memory. Recompilation reads only authorised claims.

2. Utility-ranked candidates (query time):
       u(c|x,t) = α·s_visual(x,c) + β·s_int(c | top visual candidate) + γ·p_protect(c,t)
   s_visual is min-max normalised over admitted classes; s_int is the scanned
   interference with the top-1 visual class; p_protect is a DEVELOPMENT-data prior
   (dev confusion rate from the previous episode), never computed from test errors.
   The top-1 visual class is always kept.

3. Query-contrastive packing: claims of each candidate are re-ranked by
   distinctiveness against the OTHER candidates in this query, then packed with the
   same strict b_KB cap, coverage-first.
"""
from __future__ import annotations

from ..contracts import Claim
from ..retrieval.selector import Interference, scan_interference
from .base import round_robin
from .policy import MemoryPolicy, Query, Selection


class Prism(MemoryPolicy):
    name = "prism"
    persistent = True

    def __init__(self, cfg, meter, seed: int = 0):
        super().__init__(cfg, meter, seed)
        p = cfg.get("prism", {})
        self.alpha = float(p.get("alpha", 1.0))
        self.beta = float(p.get("beta", 0.5))
        self.gamma = float(p.get("gamma", 0.25))
        self.refresh_top_k = int(p.get("refresh_top_k", 3))
        self.threshold = float(p.get("interference_threshold", 0.15))
        self.interference: dict[str, list[Interference]] = {}
        self.protect_prior: dict[str, float] = {}

    # ablation switches (set by the method factory)
    use_scan = True
    use_protect = True

    def _update(self, akb, new_classes, budget):
        assert self.index is not None
        active = sorted(akb.scope.allowed_class_ids)
        for c in sorted(new_classes):
            self._compile(c, budget, "admission")
        refreshed: set[str] = set()
        if self.use_scan:
            for c in sorted(new_classes):
                olds = [o for o in active if o not in new_classes]
                scan = scan_interference(self.index, akb, c, olds)
                for it in scan[: self.refresh_top_k]:
                    if it.score >= self.threshold and it.other not in refreshed:
                        self._compile(it.other, budget, f"protect_refresh:{c}")
                        refreshed.add(it.other)
        # full interference table among admitted classes (for s_int at query time)
        self.interference = {c: scan_interference(self.index, akb, c, active) for c in active}
        self.state.log.append({"episode": akb.scope.episode, "event": "refreshed",
                               "classes": sorted(refreshed)})

    def set_protect_prior(self, prior: dict[str, float]) -> None:
        """Dev-split confusion rates (per admitted class) from the previous episode."""
        self.protect_prior = dict(prior)

    def _s_int(self, c: str, top: str) -> float:
        for it in self.interference.get(top, []):
            if it.other == c:
                return it.score
        return 0.0

    def candidates(self, q: Query, budget):
        active = sorted(q.scope.allowed_class_ids)
        vs = q.visual_scores
        if not vs:
            return active
        lo, hi = min(vs.values()), max(vs.values())
        norm = {c: (vs.get(c, lo) - lo) / (hi - lo) if hi > lo else 0.0 for c in active}
        top = max(active, key=lambda c: (norm[c], c))
        prot = self.protect_prior if self.use_protect else {}
        u = {c: self.alpha * norm[c] + self.beta * self._s_int(c, top) + self.gamma * prot.get(c, 0.0)
             for c in active}
        u[top] = float("inf")
        ranked = sorted(active, key=lambda c: (-u[c], c))
        return ranked[:max(1, budget.r)]

    def select(self, q: Query, budget) -> Selection:
        assert self.index is not None
        cands = self.candidates(q, budget)
        per: list[list[Claim]] = []
        for c in cands:
            others = [o for o in cands if o != c]
            cl = self.card_claims(c)
            per.append(sorted(cl, key=lambda x: (-self.index.score_claim(x, others), x.claim_id)))
        return self._pack(q, cands, round_robin(per), budget)


class PrismNoScan(Prism):
    name = "prism_noscan"
    use_scan = False


class PrismNoProtect(Prism):
    name = "prism_noprotect"
    use_protect = False
