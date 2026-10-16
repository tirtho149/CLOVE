"""clove.memory.baselines — stateless baselines and mechanism ablations."""
from __future__ import annotations

import random

from ..contracts import ClaimKind, EvidenceLine
from ..retrieval.bm25 import BM25
from .base import claim_line, round_robin
from .policy import MemoryPolicy, Query, Selection


class NamesOnly(MemoryPolicy):
    """Pretrained prior: every admitted class name offered, no KB evidence."""
    name = "names"
    uses_evidence = False

    def candidates(self, q: Query, budget):
        return sorted(q.scope.allowed_class_ids)

    def select(self, q: Query, budget) -> Selection:
        cands = self.candidates(q, budget)
        return Selection(cands, {c: [] for c in cands}, 0, 0, self.state.hash())


class Sage0(MemoryPolicy):
    """Original SAGE k=0 reference: SAGE prompt wording, ALL admitted class names, and
    the SAGE KB visual descriptions of all admitted classes (no b_KB cap — it is the
    fixed-resource reference; its tokens are still counted exactly). Sentences that
    name a non-admitted class are masked (departure from SAGE, required for leakage
    safety)."""
    name = "sage0"

    def candidates(self, q: Query, budget):
        return sorted(q.scope.allowed_class_ids)

    def select(self, q: Query, budget) -> Selection:
        assert self.akb is not None
        cands = self.candidates(q, budget)
        ev: dict[str, list[EvidenceLine]] = {}
        used = 0
        for c in cands:
            lines = []
            for cl in self.akb.claims(c):
                if cl.kind == ClaimKind.KB_DESCRIPTION:
                    n = self.meter.count(cl.text)
                    lines.append(EvidenceLine(cl.claim_id, c, cl.text, n))
                    used += n
            ev[c] = lines
        return Selection(cands, ev, used, 0, self.state.hash(), prompt_style="sage0")


class Uniform(MemoryPolicy):
    """Uniform compression: equal per-candidate share of b_KB, claims in source order."""
    name = "uniform"

    def select(self, q: Query, budget) -> Selection:
        assert self.index is not None
        cands = self.candidates(q, budget)
        share = budget.kb_tokens // max(1, len(cands))
        ranked = []
        for c in cands:
            used = 0
            for cl in self.index.by_class.get(c, ()):
                n = self.meter.count(claim_line(cl))
                if n <= budget.excerpt_tokens and used + n <= share:
                    ranked.append(cl)
                    used += n
        return self._pack(q, cands, ranked, budget)


class RandomSubset(MemoryPolicy):
    name = "random"

    def select(self, q: Query, budget) -> Selection:
        assert self.index is not None
        cands = self.candidates(q, budget)
        rng = random.Random(f"{self.seed}:{q.image_id}:{q.episode}")
        per = []
        for c in cands:
            cl = list(self.index.by_class.get(c, ()))
            rng.shuffle(cl)
            per.append(cl)
        return self._pack(q, cands, round_robin(per), budget)


class BM25Evidence(MemoryPolicy):
    """BM25 claim ranking; query = names + affected parts of the candidate set."""
    name = "bm25"

    def select(self, q: Query, budget) -> Selection:
        assert self.index is not None and self.akb is not None
        cands = self.candidates(q, budget)
        pool = [cl for c in cands for cl in self.index.by_class.get(c, ())]
        if not pool:
            return self._pack(q, cands, [], budget)
        bm = BM25({cl.claim_id: cl.text for cl in pool})
        parts = []
        for c in cands:
            r = self.akb.record(c)
            parts.append(c.replace("_", " ") + " " + " ".join(r.affected_parts if r else ()))
        query = " ".join(parts)
        per = []
        for c in cands:
            cl = [x for x in pool if x.class_id == c]
            per.append(sorted(cl, key=lambda x: (-bm.score(query, x.claim_id), x.claim_id)))
        return self._pack(q, cands, round_robin(per), budget)
