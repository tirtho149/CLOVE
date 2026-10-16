"""clove.evaluation.ood — per-shift metrics and fixed-threshold prevalence shift."""
from __future__ import annotations

import random
from collections import defaultdict

from .metrics import correct, is_known, recognition, unknown_detection


def by_shift(recs) -> dict[str, dict]:
    g = defaultdict(list)
    for r in recs:
        g[(r["shift_type"], r["severity"])].append(r)
    return {f"{k[0]}@{k[1]:g}": {"recognition": recognition(v), "unknown": unknown_detection(v)}
            for k, v in sorted(g.items())}


def prevalence_shift(recs, prevalences: list[float], seed: int = 0) -> dict[str, dict]:
    """Re-weight the SAME predictions (threshold fixed) to unknown prevalence π by
    deterministic subsampling of the larger side. Reports achieved prevalence."""
    kn = sorted([r for r in recs if is_known(r)], key=lambda r: r["key"])
    un = sorted([r for r in recs if not is_known(r)], key=lambda r: r["key"])
    out = {}
    for p in prevalences:
        if not kn or not un:
            out[f"{p:.2f}"] = {"NA": "no_eligible_unknowns" if not un else "no_known"}
            continue
        rng = random.Random(f"{seed}:{p}")
        n_un_target = int(round(p * len(kn) / (1 - p)))
        if n_un_target <= len(un):
            k_sel, u_sel = kn, rng.sample(un, n_un_target)
        else:
            n_kn = int(round(len(un) * (1 - p) / p))
            k_sel, u_sel = rng.sample(kn, min(len(kn), n_kn)), un
        sub = k_sel + u_sel
        ow = sum(1 for r in sub if correct(r))
        out[f"{p:.2f}"] = {"achieved_prevalence": len(u_sel) / max(1, len(sub)), "n": len(sub),
                           "open_world_accuracy": ow / max(1, len(sub)),
                           "unknown": unknown_detection(sub)}
    return out
