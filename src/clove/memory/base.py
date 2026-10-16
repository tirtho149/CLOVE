"""clove.memory.base — memory state, card compilation and token-bounded packing.

A *card* is the agent's derived memory for one admitted class: an ordered tuple of
source claim ids (never rewritten text) selected under `card_tokens`. Cards are
compiled deterministically from the AuthorizedKB visible at compile time, so a card
compiled at episode 0 cannot contain contrasts with classes admitted later — this
is the concrete mechanism by which persistent memory can differ from a fresh state.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..contracts import AccessScope, Claim, ClaimKind, EvidenceLine, FutureClassLeak
from ..kb.inventory import AuthorizedKB
from ..retrieval.text import TfIdf
from ..util import atomic_write_json, stable_hash

KIND_PRIOR = {ClaimKind.LOOK_ALIKE: 0.30, ClaimKind.QUOTE: 0.20,
              ClaimKind.KB_DESCRIPTION: 0.15, ClaimKind.FIELD: 0.05}
FIELD_PRIOR = {"affected_parts": 0.20}


def claim_line(claim: Claim) -> str:
    return f"[{claim.claim_id}] {claim.text}"


@dataclass(frozen=True)
class Card:
    class_id: str
    claim_ids: tuple[str, ...]
    compiled_episode: int
    competitors_hash: str
    tokens: int
    reason: str = "admission"


@dataclass
class MemoryState:
    cards: dict[str, Card] = field(default_factory=dict)
    version: int = 0
    log: list[dict] = field(default_factory=list)

    def hash(self) -> str:
        return stable_hash({k: asdict(v) for k, v in sorted(self.cards.items())})

    def to_json(self) -> dict:
        return {"version": self.version, "cards": {k: asdict(v) for k, v in self.cards.items()},
                "log": self.log, "hash": self.hash()}

    @classmethod
    def from_json(cls, d: dict) -> "MemoryState":
        cards = {k: Card(**{**v, "claim_ids": tuple(v["claim_ids"])}) for k, v in d["cards"].items()}
        return cls(cards, d["version"], list(d.get("log", [])))


def snapshot(state: MemoryState, path: str | Path) -> str:
    """Tool `snapshot()`: version derived state. Never touches the class ledger."""
    atomic_write_json(path, state.to_json())
    return state.hash()


def restore(path: str | Path, scope: AccessScope) -> MemoryState:
    """Tool `restore()`: reload derived state; refuses cards of non-admitted classes
    and reports (does not silently drop) admitted classes missing a card."""
    st = MemoryState.from_json(json.loads(Path(path).read_text()))
    for cid in st.cards:
        scope.check(cid, "restored card for")
    missing = sorted(set(scope.allowed_class_ids) - set(st.cards))
    if missing:
        st.log.append({"event": "restore_missing_cards", "classes": missing})
    return st


class ClaimIndex:
    """TF-IDF over the authorised claims, rebuilt whenever the scope changes."""

    def __init__(self, akb: AuthorizedKB):
        self.akb = akb
        self.claims: dict[str, Claim] = {}
        docs: dict[str, str] = {}
        self.by_class: dict[str, tuple[Claim, ...]] = {}
        for cid in sorted(akb.scope.allowed_class_ids):
            cl = akb.claims(cid)
            self.by_class[cid] = cl
            for c in cl:
                self.claims[c.claim_id] = c
            docs[f"class::{cid}"] = " ".join(c.text for c in cl)
        for c in self.claims.values():
            docs[c.claim_id] = c.text
        self.tf = TfIdf(docs)

    def class_sim(self, a: str, b: str) -> float:
        return self.tf.sim(f"class::{a}", f"class::{b}")

    def distinctiveness(self, claim: Claim, competitors: list[str]) -> float:
        if not competitors:
            return 1.0
        v = self.tf.vec[claim.claim_id]
        return 1.0 - max(TfIdf.cos(v, self.tf.vec[f"class::{o}"]) for o in competitors)

    def score_claim(self, claim: Claim, competitors: list[str]) -> float:
        s = KIND_PRIOR.get(claim.kind, 0.0) + FIELD_PRIOR.get(claim.field, 0.0)
        s += self.distinctiveness(claim, competitors)
        if claim.mentions and set(claim.mentions) & set(competitors):
            s += 0.30      # documented, source-cited contrast with an active competitor
        return s


def compile_card(index: ClaimIndex, class_id: str, competitors: list[str], card_tokens: int,
                 excerpt_tokens: int, meter, episode: int, reason: str = "admission") -> Card:
    """Tool `compile_memory(B, policy)` for one class: immutable claims -> bounded view."""
    index.akb.scope.check(class_id, "compile_card for")
    claims = index.by_class.get(class_id, ())
    ranked = sorted(claims, key=lambda c: (-index.score_claim(c, competitors), c.claim_id))
    kept, used = [], 0
    for c in ranked:
        n = meter.count(claim_line(c))
        if n > excerpt_tokens:
            continue           # never truncate a cited claim
        if used + n <= card_tokens:
            kept.append(c.claim_id)
            used += n
    return Card(class_id, tuple(kept), episode, stable_hash(sorted(competitors)), used, reason)


def pack_evidence(ranked: list[Claim], scope: AccessScope, meter, max_tokens: int,
                  excerpt_tokens: int) -> tuple[list[EvidenceLine], int, int]:
    """Protocol §11 reference packing. Drops whole lower-ranked claims, never truncates.
    Returns (kept lines, tokens used, number of claims dropped for budget)."""
    kept: list[EvidenceLine] = []
    used = dropped = 0
    for claim in ranked:
        if claim.class_id not in scope.allowed_class_ids:
            raise FutureClassLeak("future-class KB leak")
        for m in claim.mentions:
            if m not in scope.allowed_class_ids:
                raise FutureClassLeak(f"claim {claim.claim_id} mentions non-admitted {m}")
        line = claim_line(claim)
        cost = meter.count(line)
        if cost > excerpt_tokens:
            dropped += 1
            continue
        if used + cost <= max_tokens:
            kept.append(EvidenceLine(claim.claim_id, claim.class_id, line, cost))
            used += cost
        else:
            dropped += 1
    if used > max_tokens:
        raise AssertionError("strict cap violated")
    return kept, used, dropped


def round_robin(per_class: list[list[Claim]]) -> list[Claim]:
    """Interleave ranked claim lists so every candidate gets coverage before any gets
    a second claim (rank order of classes is preserved within each round)."""
    out: list[Claim] = []
    i = 0
    while any(i < len(x) for x in per_class):
        for x in per_class:
            if i < len(x):
                out.append(x[i])
        i += 1
    return out
