"""clove.memory.fresh — fresh-state reconstruction (+ oracle and shuffled controls)."""
from __future__ import annotations

import random

from ..contracts import EvidenceLine
from .base import MemoryState, round_robin
from .policy import MemoryPolicy, Query, Selection


class FreshState(MemoryPolicy):
    """M_t^fresh: discard all derived state and rebuild every card from the authorised
    KB at each episode, with the same card/retrieval/token budgets as persistent
    methods. Its contrast with a persistent method isolates memory-history effects."""
    name = "fresh"
    persistent = False

    def _update(self, akb, new_classes, budget):
        self.state = MemoryState(version=self.state.version)
        for c in sorted(akb.scope.allowed_class_ids):
            self._compile(c, budget, "fresh_rebuild")


class OracleCandidates(FreshState):
    """Diagnostic ceiling: for KNOWN images the true class is forced into the
    candidate set (replacing the lowest-ranked). Separates retriever from recognizer.
    Never used for unknown images (oracle_class is None for them)."""
    name = "oracle"

    def candidates(self, q: Query, budget):
        c = super().candidates(q, budget)
        if q.oracle_class and q.oracle_class not in c:
            q.scope.check(q.oracle_class, "oracle candidate")
            c = c[:-1] + [q.oracle_class] if len(c) >= budget.r else c + [q.oracle_class]
        return c


class ShuffledEvidence(FreshState):
    """Negative control: identical packed evidence, but each candidate's block is
    shown under a different candidate (derangement). Matched length by construction."""
    name = "shuffled"

    def select(self, q: Query, budget) -> Selection:
        sel = super().select(q, budget)
        cands = sel.candidates
        if len(cands) < 2:
            return sel
        rng = random.Random(f"{self.seed}:{q.image_id}:{q.episode}")
        perm = cands[:]
        while any(a == b for a, b in zip(perm, cands)):
            rng.shuffle(perm)
        ev: dict[str, list[EvidenceLine]] = {}
        for src, dst in zip(cands, perm):
            ev[dst] = [EvidenceLine(l.claim_id, l.class_id, l.text, l.tokens) for l in sel.evidence.get(src, [])]
        sel.evidence = ev
        sel.notes["shuffled_from"] = dict(zip(perm, cands))
        return sel


__all__ = ["FreshState", "OracleCandidates", "ShuffledEvidence", "round_robin"]
