"""clove.evaluation.continual — acquisition, retention, forgetting (accuracy, memory,
compute). All functions take prediction dicts for ONE (order, budget) stream unless
stated otherwise, on test/original-image records."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .metrics import correct, is_known, macro_accuracy


def episodes(recs) -> list[int]:
    return sorted({r["episode"] for r in recs})


def acc_matrix(recs) -> dict[int, dict[str, float]]:
    """a[t][c] = open-world accuracy on images of admitted class c at episode t."""
    by: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if is_known(r):
            by[r["episode"]][r["gt_class"]].append(int(correct(r)))
    return {t: {c: float(np.mean(v)) for c, v in d.items()} for t, d in by.items()}


def admit_episode(recs) -> dict[str, int]:
    out = {}
    for r in recs:
        if r.get("gt_admit_episode") is not None:
            out[r["gt_class"]] = r["gt_admit_episode"]
    return out


def trajectories(recs) -> dict[int, dict]:
    """Three separate trajectories per episode: old-class, newly admitted, and
    not-yet-admitted unknown rejection."""
    out = {}
    for t in episodes(recs):
        rt = [r for r in recs if r["episode"] == t]
        old = [r for r in rt if r["gt_status"] == "old"]
        new = [r for r in rt if r["gt_status"] == "new"]
        pend = [r for r in rt if r["gt_status"] == "pending_unknown"]
        perm = [r for r in rt if r["gt_status"] == "permanent_unknown"]
        out[t] = {
            "old_macro_acc": macro_accuracy(old), "n_old": len(old),
            "new_macro_acc": macro_accuracy(new), "n_new": len(new),
            "pending_unknown_rejection": float(np.mean([r["decision"] != "KNOWN" for r in pend])) if pend else None,
            "permanent_unknown_rejection": float(np.mean([r["decision"] != "KNOWN" for r in perm])) if perm else None,
            "n_pending": len(pend), "n_permanent": len(perm),
            "no_eligible_unknowns": not pend and not perm,
        }
    return out


def acquisition(recs) -> dict[str, dict]:
    """Q_pre(c) = P(⊥ | y=c, t < t_c) and Q_post(c) = P(ŷ=c | y=c, t ≥ t_c): two
    different endpoints, reported side by side, never subtracted."""
    te = admit_episode(recs)
    out = {}
    for c, tc in te.items():
        pre = [r["decision"] != "KNOWN" for r in recs if r["gt_class"] == c and r["episode"] < tc]
        post = [r["decision"] == "KNOWN" and r["label_id"] == c for r in recs if r["gt_class"] == c and r["episode"] >= tc]
        at = [r["decision"] == "KNOWN" and r["label_id"] == c for r in recs if r["gt_class"] == c and r["episode"] == tc]
        out[c] = {"t_c": tc, "Q_pre": float(np.mean(pre)) if pre else None,
                  "Q_post": float(np.mean(post)) if post else None,
                  "acc_at_admission": float(np.mean(at)) if at else None}
    return out


def forgetting(recs) -> dict:
    """F_c = max_{τ∈[t_c,T]} a_{τ,c} − a_{T,c}; BWT = mean_c (a_{T,c} − a_{t_c,c}) over
    classes admitted before T."""
    a = acc_matrix(recs)
    if not a:
        return {}
    T = max(a)
    te = admit_episode(recs)
    Fc, bwt = {}, []
    for c, tc in te.items():
        hist = [a[t][c] for t in range(tc, T + 1) if t in a and c in a[t]]
        if not hist or c not in a.get(T, {}):
            continue
        Fc[c] = max(hist) - a[T][c]
        if tc < T and c in a.get(tc, {}):
            bwt.append(a[T][c] - a[tc][c])
    vals = list(Fc.values())
    return {"per_class": Fc, "mean": float(np.mean(vals)) if vals else None,
            "worst": max(vals) if vals else None, "bwt": float(np.mean(bwt)) if bwt else None,
            "final_episode": T}


def expansion_forgetting(recs) -> dict[int, float | None]:
    """F_exp(t) = A(D_old; C_0, M_0) − A(D_old; C_t, M_t), D_old = images of C_0."""
    c0 = {r["gt_class"] for r in recs if r["episode"] == 0 and r["gt_status"] == "new"}
    base = macro_accuracy([r for r in recs if r["episode"] == 0 and r["gt_class"] in c0])
    out = {}
    for t in episodes(recs):
        at = macro_accuracy([r for r in recs if r["episode"] == t and r["gt_class"] in c0])
        out[t] = None if base is None or at is None else base - at
    return out


def paired_old(recs_a, recs_b, t: int) -> list[tuple[dict, dict]]:
    """Pair two methods' records on identical (image, variant) old-class items at t."""
    ia = {(r["image_id"], r["variant_id"]): r for r in recs_a if r["episode"] == t and r["gt_status"] == "old"}
    ib = {(r["image_id"], r["variant_id"]): r for r in recs_b if r["episode"] == t and r["gt_status"] == "old"}
    return [(ia[k], ib[k]) for k in sorted(set(ia) & set(ib))]


def memory_forgetting(fresh, persistent) -> dict[int, dict]:
    """F_memory(t) = A(D_old; C_t, M_t^fresh) − A(D_old; C_t, M_t^persistent) on paired
    images (same inventory, retrieval and token caps by construction). Positive =
    persistent memory is worse than a fresh rebuild."""
    out = {}
    for t in episodes(fresh):
        pairs = paired_old(fresh, persistent, t)
        if not pairs:
            continue
        af = macro_accuracy([a for a, _ in pairs])
        ap = macro_accuracy([b for _, b in pairs])
        out[t] = {"F_memory": None if af is None or ap is None else af - ap, "n_pairs": len(pairs),
                  "acc_fresh": af, "acc_persistent": ap}
    return out


def compute_forgetting(recs) -> dict:
    """F_tok(j,t) = (T_{j,t} − T_{j,t_j}) / max(1, T_{j,t_j}) for old-class images j,
    T = prompt+output tokens (end-to-end, all calls). Reported both conditional on the
    item being correct at t_j and at t, and unconditionally, plus component growth."""
    by = defaultdict(dict)
    for r in recs:
        if is_known(r):
            by[(r["image_id"], r["variant_id"])][r["episode"]] = r
    cond, uncond = defaultdict(list), defaultdict(list)
    comp = defaultdict(lambda: defaultdict(list))
    for item in by.values():
        tj = min(item)
        base = item[tj]
        T0 = base["prompt_tokens"] + base["output_tokens"]
        for t, r in item.items():
            if t == tj:
                continue
            Tt = r["prompt_tokens"] + r["output_tokens"]
            f = (Tt - T0) / max(1, T0)
            uncond[t].append(f)
            if correct(base) and correct(r):
                cond[t].append(f)
            comp[t]["prompt"].append(r["prompt_tokens"] - base["prompt_tokens"])
            comp[t]["output"].append(r["output_tokens"] - base["output_tokens"])
            comp[t]["kb"].append(r["kb_tokens"] - base["kb_tokens"])
            comp[t]["calls"].append(r["vlm_calls"] - base["vlm_calls"])
            comp[t]["latency"].append(r["latency_s"] - base["latency_s"])
    return {t: {"F_tok_correct_mean": float(np.mean(cond[t])) if cond[t] else None,
                "n_correct_pairs": len(cond[t]),
                "F_tok_all_mean": float(np.mean(uncond[t])) if uncond[t] else None,
                "n_all": len(uncond[t]),
                "delta": {k: float(np.mean(v)) for k, v in comp[t].items()}}
            for t in sorted(uncond)}
