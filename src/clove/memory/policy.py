"""clove.memory.policy — the MemoryPolicy interface shared by every method."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Budget
from ..contracts import AccessScope, Claim, EvidenceLine
from ..kb.inventory import AuthorizedKB
from .base import ClaimIndex, MemoryState, compile_card, pack_evidence, round_robin


@dataclass
class Query:
    image_id: str
    episode: int
    scope: AccessScope
    visual_scores: dict[str, float]          # admitted classes only
    oracle_class: str | None = None          # evaluator-only; used ONLY by `oracle`


@dataclass
class Selection:
    candidates: list[str]                    # option order shown to the model
    evidence: dict[str, list[EvidenceLine]]  # class -> lines
    kb_tokens: int
    dropped_claims: int
    memory_state_hash: str
    prompt_style: str = "clove"              # clove | sage0
    notes: dict = field(default_factory=dict)


class MemoryPolicy:
    name = "base"
    persistent = False          # does state carry over between episodes?
    uses_evidence = True

    def __init__(self, cfg, meter, seed: int = 0):
        self.cfg = cfg
        self.meter = meter
        self.seed = seed
        self.state = MemoryState()
        self.index: ClaimIndex | None = None
        self.akb: AuthorizedKB | None = None
        self.card_tokens = int(cfg.get("memory.card_tokens", 128))

    # ── lifecycle ──────────────────────────────────────────────────────────
    def update(self, akb: AuthorizedKB, new_classes: tuple[str, ...], budget: Budget) -> None:
        """Called once per episode after admission with the new AuthorizedKB."""
        self.akb = akb
        self.index = ClaimIndex(akb)
        for c in new_classes:
            akb.scope.check(c, "update with")
        self._update(akb, new_classes, budget)
        self.state.version += 1

    def _update(self, akb, new_classes, budget) -> None:
        pass

    def _compile(self, cid: str, budget: Budget, reason: str) -> None:
        assert self.index is not None and self.akb is not None
        comps = [c for c in sorted(self.akb.scope.allowed_class_ids) if c != cid]
        self.state.cards[cid] = compile_card(self.index, cid, comps, self.card_tokens,
                                             budget.excerpt_tokens, self.meter,
                                             self.akb.scope.episode, reason)
        self.state.log.append({"episode": self.akb.scope.episode, "event": reason, "class": cid})

    # ── selection ──────────────────────────────────────────────────────────
    def candidates(self, q: Query, budget: Budget) -> list[str]:
        from ..retrieval.selector import retrieve_active
        return retrieve_active(q.visual_scores, q.scope, budget.r)

    def card_claims(self, cid: str) -> list[Claim]:
        assert self.index is not None
        card = self.state.cards.get(cid)
        if card is None:
            return []
        return [self.index.claims[i] for i in card.claim_ids if i in self.index.claims]

    def select(self, q: Query, budget: Budget) -> Selection:
        cands = self.candidates(q, budget)
        ranked = round_robin([self.card_claims(c) for c in cands])
        return self._pack(q, cands, ranked, budget)

    def _pack(self, q: Query, cands: list[str], ranked: list[Claim], budget: Budget,
              **notes) -> Selection:
        lines, used, dropped = pack_evidence(ranked, q.scope, self.meter, budget.kb_tokens,
                                             budget.excerpt_tokens)
        ev: dict[str, list[EvidenceLine]] = {c: [] for c in cands}
        for ln in lines:
            ev.setdefault(ln.class_id, []).append(ln)
        return Selection(cands, ev, used, dropped, self.state.hash(), notes=notes)
