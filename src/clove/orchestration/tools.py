"""clove.orchestration.tools — the eight protocol tools as typed, pure(-ish) functions.

These are thin, test-covered entry points over the implementing modules so that the
tool contract in protocol §6 maps 1:1 onto code:

  admit_class      -> kb.inventory.ClassLedger.admit
  retrieve_active  -> retrieval.selector.retrieve_active
  scan_interference-> retrieval.selector.scan_interference
  compile_memory   -> memory.base.compile_card (per class) / MemoryPolicy.update
  gate_unknown     -> orchestration.policy.UnknownGate.decide
  snapshot/restore -> memory.base.snapshot / restore
  budget_meter     -> inference.token_meter.BudgetMeter
  audit_episode    -> below
"""
from __future__ import annotations

from ..contracts import AccessScope, FutureClassLeak, assert_in_scope
from ..inference.token_meter import BudgetMeter
from ..kb.inventory import ClassLedger
from ..memory.base import compile_card, restore, snapshot
from ..memory.policy import Selection
from ..orchestration.policy import UnknownGate
from ..retrieval.selector import retrieve_active, scan_interference


def admit_class(ledger: ClassLedger, class_id: str, episode: int) -> AccessScope:
    ledger.admit(class_id, episode)
    return ledger.scope()


def gate_unknown(gate: UnknownGate, decoded):
    return gate.decide(decoded)


def budget_meter(tokenizer) -> BudgetMeter:
    return BudgetMeter(tokenizer)


def audit_episode(scope: AccessScope, ledger: ClassLedger, selection: Selection,
                  label_id: str | None, index_claims: dict, manifest_fp: str,
                  expected_manifest_fp: str) -> list[str]:
    """Verifies allowed class ids, citations, hashes and manifest for one prediction.
    Returns a list of violations (empty == pass); FUTURE_LEAKAGE raises."""
    problems: list[str] = []
    if set(scope.allowed_class_ids) != set(ledger.admitted):
        raise FutureClassLeak("scope differs from ledger")
    for c in selection.candidates:
        scope.check(c, "candidate")
    claims = []
    for cid, lines in selection.evidence.items():
        scope.check(cid, "evidence block")
        for ln in lines:
            cl = index_claims.get(ln.claim_id)
            if cl is None:
                problems.append(f"CITATION_MISMATCH:{ln.claim_id}")
                continue
            claims.append(cl)
            if selection.prompt_style == "clove" and not ln.text.startswith(f"[{ln.claim_id}]"):
                problems.append(f"CITATION_MISMATCH:{ln.claim_id}")
    assert_in_scope(scope, claims, label_id)
    if manifest_fp != expected_manifest_fp:
        problems.append("MANIFEST_FINGERPRINT_CHANGED")
    return problems


__all__ = ["admit_class", "retrieve_active", "scan_interference", "compile_card",
           "gate_unknown", "snapshot", "restore", "budget_meter", "audit_episode"]
