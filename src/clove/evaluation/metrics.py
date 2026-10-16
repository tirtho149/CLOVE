"""clove.evaluation.metrics — open-world recognition and unknown-detection metrics.

Conventions (fixed in advance):
* A rejection (⊥) of a KNOWN image is an error for open-world accuracy.
* Unknown score s: higher = more likely unknown. Missing scores (parse failures,
  errors) are set to 1.0, consistent with UNRESOLVED -> ⊥.
* Unknown-detection metrics treat UNKNOWN images as positives ("out"). They are NA
  (None), never 0, when either class is empty.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np

KNOWN = ("old", "new")
UNKNOWN = ("pending_unknown", "permanent_unknown")


def is_known(r: dict) -> bool:
    return r["gt_status"] in KNOWN


def score(r: dict) -> float:
    s = r.get("unknown_score")
    return 1.0 if s is None or (isinstance(s, float) and math.isnan(s)) else float(s)


def closed_label(r: dict) -> str | None:
    """Closed-set argmax over offered admitted classes (independent of the gate)."""
    ks = r.get("known_scores") or {}
    if ks:
        return max(sorted(ks), key=lambda c: ks[c])
    return r.get("label_id")


def correct(r: dict) -> bool:
    if is_known(r):
        return r["decision"] == "KNOWN" and r["label_id"] == r["gt_class"]
    return r["decision"] != "KNOWN"


def per_class_accuracy(recs: Iterable[dict]) -> dict[str, float]:
    by: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        if is_known(r):
            by[r["gt_class"]].append(int(correct(r)))
    return {c: float(np.mean(v)) for c, v in by.items() if v}


def macro_accuracy(recs: Iterable[dict]) -> float | None:
    pc = per_class_accuracy(recs)
    return float(np.mean(list(pc.values()))) if pc else None


def micro_accuracy(recs: Iterable[dict]) -> float | None:
    xs = [int(correct(r)) for r in recs if is_known(r)]
    return float(np.mean(xs)) if xs else None


def macro_f1(recs: Iterable[dict]) -> float | None:
    """Macro-F1 over the admitted classes present in ground truth; ⊥ counts as a
    prediction of no admitted class (it lowers recall, not any class's precision)."""
    recs = [r for r in recs if is_known(r)]
    if not recs:
        return None
    classes = sorted({r["gt_class"] for r in recs})
    f1s = []
    for c in classes:
        tp = sum(1 for r in recs if r["gt_class"] == c and r["decision"] == "KNOWN" and r["label_id"] == c)
        fp = sum(1 for r in recs if r["gt_class"] != c and r["decision"] == "KNOWN" and r["label_id"] == c)
        fn = sum(1 for r in recs if r["gt_class"] == c) - tp
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def rejection_rate(recs: Iterable[dict]) -> float | None:
    xs = [int(r["decision"] != "KNOWN") for r in recs if not is_known(r)]
    return float(np.mean(xs)) if xs else None


# ── threshold-free unknown detection ────────────────────────────────────────

def auroc(pos: list[float], neg: list[float]) -> float | None:
    """P(score_pos > score_neg) + 0.5 P(tie) (Mann–Whitney)."""
    if not pos or not neg:
        return None
    allv = np.array(neg + pos)
    ranks = _rankdata(allv)
    rp = ranks[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def average_precision(pos: list[float], neg: list[float]) -> float | None:
    """AP with positives = `pos` (higher score = more positive); step-wise, ties
    broken pessimistically (negatives first)."""
    if not pos:
        return None
    items = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: (-x[0], x[1]))
    tp = 0
    ap = 0.0
    for k, (_, y) in enumerate(items, 1):
        if y:
            tp += 1
            ap += tp / k
    return ap / len(pos)


def fpr_at_tpr(known: list[float], unknown: list[float], tpr: float = 0.95) -> float | None:
    """Known = in-distribution, accepted when s <= τ. τ chosen so >= tpr of known are
    accepted; FPR = fraction of unknown also accepted."""
    if not known or not unknown:
        return None
    ks = np.sort(np.array(known))
    tau = ks[max(0, min(len(ks) - 1, math.ceil(tpr * len(ks)) - 1))]
    return float(np.mean(np.array(unknown) <= tau))


def oscr(known_recs: list[dict], unknown_recs: list[dict]) -> float | None:
    """Area under CCR vs FPR as the acceptance threshold sweeps (Dhamija et al.)."""
    if not known_recs or not unknown_recs:
        return None
    ks = [(score(r), int(closed_label(r) == r["gt_class"])) for r in known_recs]
    us = sorted(score(r) for r in unknown_recs)
    thr = sorted({s for s, _ in ks} | set(us))
    pts = [(0.0, 0.0)]
    for th in thr:
        ccr = sum(c for s, c in ks if s <= th) / len(ks)
        fpr = sum(1 for s in us if s <= th) / len(us)
        pts.append((fpr, ccr))
    pts.append((1.0, pts[-1][1]))
    pts.sort()
    return float(sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(pts, pts[1:])))


def aurc(known_recs: list[dict]) -> float | None:
    """Area under risk–coverage on KNOWN images ordered by confidence (low s first).
    Risk = closed-set label error among the accepted."""
    if not known_recs:
        return None
    xs = sorted(known_recs, key=score)
    err = 0
    risks = []
    for k, r in enumerate(xs, 1):
        err += int(closed_label(r) != r["gt_class"])
        risks.append(err / k)
    return float(np.mean(risks))


def unknown_detection(recs: list[dict]) -> dict:
    kn = [r for r in recs if is_known(r)]
    un = [r for r in recs if not is_known(r)]
    ks, us = [score(r) for r in kn], [score(r) for r in un]
    f = fpr_at_tpr(ks, us)
    return {
        "n_known": len(kn), "n_unknown": len(un),
        "auroc": auroc(us, ks),
        "aupr_out": average_precision(us, ks),
        "aupr_in": average_precision([-s for s in ks], [-s for s in us]),
        "fpr_at_95tpr": f, "tnr_at_95tpr": None if f is None else 1 - f,
        "oscr": oscr(kn, un), "aurc": aurc(kn),
        "rejection_rate_unknown": rejection_rate(un),
        "no_eligible_unknowns": len(un) == 0,
    }


def recognition(recs: list[dict]) -> dict:
    kn = [r for r in recs if is_known(r)]
    return {"n": len(kn), "macro_accuracy": macro_accuracy(kn), "micro_accuracy": micro_accuracy(kn),
            "balanced_accuracy": macro_accuracy(kn), "macro_f1": macro_f1(kn),
            "per_class": per_class_accuracy(kn),
            "false_reject_rate": float(np.mean([r["decision"] != "KNOWN" for r in kn])) if kn else None}


def efficiency(recs: list[dict]) -> dict:
    if not recs:
        return {}
    tin = np.array([r["prompt_tokens"] for r in recs], dtype=float)
    tout = np.array([r["output_tokens"] for r in recs], dtype=float)
    kb = np.array([r["kb_tokens"] for r in recs], dtype=float)
    tot = tin + tout
    ncor = sum(int(correct(r)) for r in recs)
    return {
        "n": len(recs), "n_correct": ncor,
        "prompt_tokens": _summ(tin), "output_tokens": _summ(tout), "kb_tokens": _summ(kb),
        "total_tokens": _summ(tot), "vlm_calls": float(np.sum([r["vlm_calls"] for r in recs])),
        "latency_s": _summ(np.array([r["latency_s"] for r in recs], dtype=float)),
        "tokens_per_correct": float(tot.sum() / max(1, ncor)),
        "correct_per_1M_input_tokens": float(ncor / max(1.0, tin.sum()) * 1e6),
        "image_tokens_exposed": all(r.get("image_tokens") is not None for r in recs),
        "usd_incurred": float(sum((r.get("extra") or {}).get("cost_incurred") or 0.0 for r in recs)),
        "usd_list_equivalent": float(sum((r.get("extra") or {}).get("cost_usd") or 0.0 for r in recs)),
        "usd_per_correct": float(sum((r.get("extra") or {}).get("cost_usd") or 0.0 for r in recs) / max(1, ncor)),
    }


def _summ(x: np.ndarray) -> dict:
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "total": float(x.sum())}


def failure_breakdown(recs: list[dict]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for r in recs:
        out[r["failure"]] += 1
    return dict(out)
