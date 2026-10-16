"""clove.kb.inventory — the authorised class ledger and the scoped KB view.

The ledger is the ONLY object that can widen access: `admit()` adds classes from the
pre-committed schedule and returns a new `AccessScope`. Every downstream module gets
an `AuthorizedKB` built from that scope; it physically cannot return claims of a class
that is not admitted, nor claims that *mention* a non-admitted class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..contracts import AccessScope, Claim, ClassRecord, FutureClassLeak
from ..util import stable_hash
from .reader import KBRegistry


def inventory_hash(class_ids) -> str:
    return stable_hash(sorted(class_ids))


@dataclass
class ClassLedger:
    schedule: list[list[str]]            # schedule[t] = classes admitted at episode t
    episode: int = -1
    admitted: dict[str, int] = field(default_factory=dict)   # class -> admit episode
    history: list[dict] = field(default_factory=list)

    @property
    def all_scheduled(self) -> set[str]:
        return {c for b in self.schedule for c in b}

    def admit(self, class_id: str, episode: int) -> None:
        """Tool `admit_class(id, episode)`: only ids scheduled for THIS episode."""
        if episode >= len(self.schedule) or class_id not in self.schedule[episode]:
            raise FutureClassLeak(f"{class_id!r} is not authorised for episode {episode}")
        if class_id in self.admitted:
            return
        self.admitted[class_id] = episode
        self.history.append({"episode": episode, "class_id": class_id})

    def advance(self) -> AccessScope:
        """Move to the next episode and admit its whole batch."""
        nxt = self.episode + 1
        if nxt >= len(self.schedule):
            raise IndexError("schedule exhausted")
        for c in self.schedule[nxt]:
            self.admit(c, nxt)
        self.episode = nxt
        return self.scope()

    def scope(self) -> AccessScope:
        ids = frozenset(self.admitted)
        return AccessScope(self.episode, ids, inventory_hash(ids))

    def admit_episode_of(self, class_id: str) -> int | None:
        for t, batch in enumerate(self.schedule):
            if class_id in batch:
                return t
        return None


class AuthorizedKB:
    """Scope-enforcing view over the immutable registry."""

    def __init__(self, registry: KBRegistry, scope: AccessScope, mask_mentions: bool = True):
        self._reg = registry
        self.scope = scope
        self.mask_mentions = mask_mentions
        self.masked_claims: dict[str, int] = {}

    def claims(self, class_id: str) -> tuple[Claim, ...]:
        self.scope.check(class_id, "KB request for")
        out = []
        masked = 0
        for c in self._reg._get_claims(class_id):
            if any(m not in self.scope.allowed_class_ids for m in c.mentions):
                if self.mask_mentions:
                    masked += 1
                    continue
                raise FutureClassLeak(f"claim {c.claim_id} names a non-admitted class")
            out.append(c)
        self.masked_claims[class_id] = masked
        return tuple(out)

    def record(self, class_id: str) -> ClassRecord | None:
        self.scope.check(class_id, "record request for")
        r = self._reg._get_record(class_id)
        if r is None:
            return None
        # look-alikes are filtered to the admitted inventory
        la = tuple(x for x in r.look_alikes if x in self.scope.allowed_class_ids)
        return ClassRecord(r.class_id, r.display_name, r.aliases, r.disease_type,
                           r.affected_parts, r.pathogen, la)

    def all_active_claims(self) -> dict[str, tuple[Claim, ...]]:
        return {c: self.claims(c) for c in sorted(self.scope.allowed_class_ids)}
