"""clove.kb.provenance — citation checks and the source/licence ledger."""
from __future__ import annotations

from collections import Counter
from urllib.parse import urlparse

from ..contracts import Claim, ClaimKind
from .reader import KBRegistry


def citation_problems(claim: Claim) -> list[str]:
    """Static checks that need no network. Re-verifying quotes against live pages is
    deliberately out of scope (SAGE already did it; CLOVE must not crawl)."""
    probs = []
    if claim.kind in (ClaimKind.QUOTE, ClaimKind.FIELD, ClaimKind.LOOK_ALIKE):
        if not claim.source_url:
            probs.append("missing_url")
        elif urlparse(claim.source_url).scheme not in ("http", "https"):
            probs.append("bad_url_scheme")
        if not claim.verbatim_quote:
            probs.append("missing_quote")
    if claim.kind == ClaimKind.QUOTE and claim.field == "provenance":
        if claim.text != claim.verbatim_quote:
            probs.append("text_differs_from_quote")
    return probs


def provenance_ledger(reg: KBRegistry) -> dict:
    """Per-class counts of claim kinds, citation problems and source domains."""
    out = {}
    for cid in reg.class_ids:
        claims = reg._get_claims(cid)
        domains = Counter(urlparse(c.source_url).netloc for c in claims if c.source_url)
        problems = Counter(p for c in claims for p in citation_problems(c))
        out[cid] = {
            "n_claims": len(claims),
            "kinds": dict(Counter(c.kind.value for c in claims)),
            "n_with_url_and_quote": sum(1 for c in claims if c.source_url and c.verbatim_quote),
            "domains": dict(domains),
            "problems": dict(problems),
            "mentions_other_classes": sum(1 for c in claims if c.mentions),
        }
    return out
