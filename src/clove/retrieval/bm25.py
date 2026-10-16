"""clove.retrieval.bm25 — Okapi BM25 over claim texts (mechanism ablation)."""
from __future__ import annotations

import math
from collections import Counter

from .text import tokens


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = {k: Counter(tokens(v)) for k, v in docs.items()}
        self.len = {k: sum(c.values()) for k, c in self.tf.items()}
        self.avg = (sum(self.len.values()) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        for c in self.tf.values():
            df.update(c.keys())
        n = len(self.tf)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str, key: str) -> float:
        tf, L = self.tf[key], self.len[key] or 1
        s = 0.0
        for t in tokens(query):
            f = tf.get(t, 0)
            if f:
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / (self.avg or 1)))
        return s
