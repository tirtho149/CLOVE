"""clove.memory.append — persistent append-only memory (+ recency ablation)."""
from __future__ import annotations

from ..contracts import Claim
from .base import round_robin
from .policy import MemoryPolicy, Query, Selection


class AppendOnly(MemoryPolicy):
    """Each class's card is compiled ONCE, at admission, against the classes active
    then; it is never revisited. Retrieval caps identical to every other method."""
    name = "append"
    persistent = True

    def _update(self, akb, new_classes, budget):
        for c in sorted(new_classes):
            self._compile(c, budget, "admission")


class Recency(AppendOnly):
    """Mechanism ablation: same cards, but evidence budget goes to the most recently
    admitted candidates first."""
    name = "recency"

    def select(self, q: Query, budget) -> Selection:
        cands = self.candidates(q, budget)
        order = sorted(cands, key=lambda c: (-self.state.cards[c].compiled_episode
                                             if c in self.state.cards else 0, cands.index(c)))
        ranked: list[Claim] = []
        for c in order:
            ranked += self.card_claims(c)
        return self._pack(q, cands, ranked, budget)


__all__ = ["AppendOnly", "Recency", "round_robin"]
