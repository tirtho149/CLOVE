"""clove.retrieval.selector — `retrieve_active` and `scan_interference` tools."""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts import AccessScope
from ..kb.inventory import AuthorizedKB
from .text import jaccard


@dataclass(frozen=True)
class Interference:
    other: str
    score: float
    text_sim: float
    organ_overlap: float
    documented: bool
    evidence: tuple[str, ...]     # claim ids / record fields supporting the score


def scan_interference(index, akb: AuthorizedKB, class_id: str, others: list[str]
                      ) -> list[Interference]:
    """Source-claim overlap + documented contrasts between `class_id` and admitted
    `others`. Uses only fields that exist; a field missing from one class is treated
    as *unknown*, never as evidence that the symptom is absent."""
    akb.scope.check(class_id, "scan_interference for")
    rec = akb.record(class_id)
    out = []
    for o in others:
        if o == class_id:
            continue
        akb.scope.check(o, "scan_interference against")
        ro = akb.record(o)
        ts = index.class_sim(class_id, o)
        if rec and ro and rec.affected_parts and ro.affected_parts:
            org = jaccard(set(rec.affected_parts), set(ro.affected_parts))
        else:
            org = 0.0
        doc_ids = [c.claim_id for c in index.by_class.get(class_id, ()) if o in c.mentions]
        doc_ids += [c.claim_id for c in index.by_class.get(o, ()) if class_id in c.mentions]
        documented = bool(doc_ids) or bool(rec and o in rec.look_alikes) or bool(ro and class_id in ro.look_alikes)
        score = 0.5 * ts + 0.3 * org + 0.2 * float(documented)
        out.append(Interference(o, score, ts, org, documented, tuple(sorted(doc_ids))))
    return sorted(out, key=lambda x: (-x.score, x.other))


def retrieve_active(visual_scores: dict[str, float] | None, scope: AccessScope, r: int
                    ) -> list[str]:
    """Tool `retrieve_active`: top-r ADMITTED classes by image-text score. With no
    visual retriever, every admitted class is a candidate (r is then not binding)."""
    active = sorted(scope.allowed_class_ids)
    if not visual_scores:
        return active
    for c in visual_scores:
        if c not in scope.allowed_class_ids:
            raise PermissionError(f"retriever returned non-admitted class {c}")
    ranked = sorted(active, key=lambda c: (-visual_scores.get(c, float("-inf")), c))
    return ranked[:max(1, r)]
