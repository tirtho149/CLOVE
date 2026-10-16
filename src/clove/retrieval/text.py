"""clove.retrieval.text — tiny deterministic text utilities (no external models)."""
from __future__ import annotations

import math
import re
from collections import Counter

_TOK = re.compile(r"[a-z][a-z\-]+")
STOP = frozenset("""
a an the and or of on in to with by for from as at is are be been being this that these those
it its may can often usually also which when where while into onto than then there their they
such other more most some any all each both not no only very will would could should has have had
do does did up down out over under between within without about across after before during per
plant plants leaf leaves soybean soybeans disease diseases symptom symptoms infected infection
""".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in STOP and len(t) > 2]


class TfIdf:
    def __init__(self, docs: dict[str, str]):
        self.docs = {k: Counter(tokens(v)) for k, v in docs.items()}
        n = max(1, len(self.docs))
        df: Counter = Counter()
        for c in self.docs.values():
            df.update(c.keys())
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        self.vec = {k: self._vec(c) for k, c in self.docs.items()}

    def _vec(self, c: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(f)) * self.idf.get(t, 1.0) for t, f in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def embed(self, text: str) -> dict[str, float]:
        return self._vec(Counter(tokens(text)))

    @staticmethod
    def cos(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(t, 0.0) for t, x in a.items())

    def sim(self, k1: str, k2: str) -> float:
        return self.cos(self.vec[k1], self.vec[k2])


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0
