"""clove.reports.render — `clove analyze`: metrics, pre-registered tests, report."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..contracts import RunStatus
from ..evaluation import continual as C
from ..evaluation import metrics as M
from ..evaluation import ood as O
from ..evaluation import statistics as S
from ..util import JsonlStore, atomic_write_json, read_json, write_status
from . import figures as F
from .tables import md_table, write_csv

PERSISTENT = ("append", "prism", "recency", "prism_noscan", "prism_noprotect")


def load(run_dir: Path) -> list[dict]:
    return list(JsonlStore(run_dir / "predictions.jsonl").iter())


def _groups(recs):
    g = defaultdict(list)
    for r in recs:
        g[(r["order_id"], r["budget_id"], r["method"])].append(r)
    return g


def primary_units(recs, treat: str, ctrl: str, budget: str, post_only: bool = True):
    """Paired old-class units (same order, image, variant, episode) for the endpoint."""
    a = {(r["order_id"], r["image_id"], r["variant_id"], r["episode"]): r for r in recs
         if r["method"] == treat and r["budget_id"] == budget and r["gt_status"] == "old"
         and r["shift_type"] == "iid"}
    b = {(r["order_id"], r["image_id"], r["variant_id"], r["episode"]): r for r in recs
         if r["method"] == ctrl and r["budget_id"] == budget and r["gt_status"] == "old"
         and r["shift_type"] == "iid"}
    units = []
    for k in sorted(set(a) & set(b)):
        if post_only and k[3] < 1:
            continue
        ra, rb = a[k], b[k]
        units.append(S.PairedUnit(ra["order_id"], ra["group_id"], ra["gt_class"],
                                  float(M.correct(ra)), float(M.correct(rb)), ra["episode"]))
    return units


def analyze(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    prereg = read_json(run_dir / "preregistration.json")
    stats_cfg = prereg["stats"]
    recs = load(run_dir)
    out_m = run_dir / "metrics"
    synthetic = any(r.get("synthetic") for r in recs)
    if not recs:
        write_status(run_dir, "analyze", RunStatus.BLOCKED, {"reason": "no predictions"})
        return {"status": "BLOCKED"}
    iid = [r for r in recs if r["shift_type"] == "iid"]

    # ── per (order, budget, method) ─────────────────────────────────────
    summary_rows, traj_rows, eff_rows = [], [], []
    per_group = {}
    for (o, b, m), rs in sorted(_groups(iid).items()):
        traj = C.trajectories(rs)
        res = {
            "recognition_by_episode": {t: M.recognition([r for r in rs if r["episode"] == t]) for t in C.episodes(rs)},
            "unknown_by_episode": {t: M.unknown_detection([r for r in rs if r["episode"] == t]) for t in C.episodes(rs)},
            "trajectories": traj, "acquisition": C.acquisition(rs), "forgetting": C.forgetting(rs),
            "expansion_forgetting": C.expansion_forgetting(rs), "compute_forgetting": C.compute_forgetting(rs),
            "efficiency": M.efficiency(rs), "failures": M.failure_breakdown(rs),
        }
        per_group[f"{o}|{b}|{m}"] = res
        for t, v in traj.items():
            traj_rows.append({"order_id": o, "budget_id": b, "method": m, "episode": t, **v})
        fg = res["forgetting"]
        e = res["efficiency"]
        summary_rows.append({
            "order_id": o, "budget_id": b, "method": m, "n": len(rs),
            "open_world_acc": float(np.mean([M.correct(r) for r in rs])),
            "known_macro_acc": M.macro_accuracy(rs),
            "unknown_auroc_mean": _mean([u["auroc"] for u in res["unknown_by_episode"].values()]),
            "forgetting_mean": fg.get("mean"), "forgetting_worst": fg.get("worst"), "bwt": fg.get("bwt"),
            "tokens_per_correct": e.get("tokens_per_correct"),
            "mean_prompt_tokens": e.get("prompt_tokens", {}).get("mean"),
            "mean_kb_tokens": e.get("kb_tokens", {}).get("mean"),
        })
        eff_rows.append({"method": m, "budget_id": b, "order_id": o,
                         "acc": float(np.mean([M.correct(r) for r in rs])),
                         "mean_total_tokens": e.get("total_tokens", {}).get("mean")})

    # ── memory-history contrast ─────────────────────────────────────────
    mem_rows = []
    by = _groups(iid)
    for (o, b, m), rs in by.items():
        if m in PERSISTENT and (o, b, "fresh") in by:
            for t, v in C.memory_forgetting(by[(o, b, "fresh")], rs).items():
                mem_rows.append({"order_id": o, "budget_id": b, "method": m, "episode": t, **v})

    # ── pre-registered tests ────────────────────────────────────────────
    pb = prereg["primary_budget"]
    tests = {}
    pvals_primary, pvals_expl = {}, {}
    fams = [("primary", stats_cfg["primary_family"]), ("exploratory", stats_cfg.get("exploratory_family", []))]
    for fam, pairs in fams:
        for treat, ctrl in pairs:
            units = primary_units(iid, treat, ctrl, pb)
            name = f"{treat}-vs-{ctrl}"
            if not units:
                tests[name] = {"family": fam, "status": "NA (methods not run or no paired old-class items)"}
                continue
            boot = S.paired_cluster_bootstrap(units, n_boot=int(stats_cfg["bootstrap"]),
                                              ci=float(stats_cfg["ci"]), seed=int(stats_cfg["seed"]))
            perm = S.sign_flip_test(units, seed=int(stats_cfg["seed"]))
            mc = S.mcnemar_exact([int(u.a) for u in units], [int(u.b) for u in units])
            tests[name] = {"family": fam, "metric": "old_class_macro_accuracy (t>=1, averaged)",
                           "bootstrap": boot, "sign_flip": perm, "mcnemar_itemwise": mc,
                           "noninferiority": S.noninferiority(boot["lo"], float(stats_cfg["noninferiority_margin"]))}
            if perm["p"] is not None:
                (pvals_primary if fam == "primary" else pvals_expl)[name] = perm["p"]
    for k, v in S.holm(pvals_primary).items():
        tests[k]["p_holm"] = v
    for k, v in S.benjamini_hochberg(pvals_expl).items():
        tests[k]["p_bh"] = v

    # ── OOD / prevalence ────────────────────────────────────────────────
    ood = {}
    for (o, b, m), rs in sorted(_groups(recs).items()):
        if b != pb:
            continue
        ood[f"{o}|{m}"] = {"by_shift": O.by_shift(rs),
                           "prevalence": O.prevalence_shift([r for r in rs if r["shift_type"] == "iid"],
                                                            prereg.get("prevalence", [0.1, 0.25, 0.5]))}

    # ── write ───────────────────────────────────────────────────────────
    out_m.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_m / "per_group.json", per_group)
    atomic_write_json(out_m / "tests.json", tests)
    atomic_write_json(out_m / "ood.json", ood)
    write_csv(out_m / "summary.csv", summary_rows)
    write_csv(out_m / "trajectories.csv", traj_rows)
    write_csv(out_m / "memory_contrast.csv", mem_rows)
    write_csv(out_m / "efficiency.csv", eff_rows)

    figs = {}
    prim_traj = _mean_over_orders([r for r in traj_rows if r["budget_id"] == pb])
    figs["F3_trajectories"] = F.trajectories(prim_traj, run_dir / "figures" / "F3_trajectories.png")
    figs["F4_memory_contrast"] = F.memory_contrast([r for r in mem_rows if r["budget_id"] == pb],
                                                   run_dir / "figures" / "F4_memory_contrast.png")
    figs["F6_accuracy_vs_tokens"] = F.accuracy_vs_tokens(_mean_eff(eff_rows), run_dir / "figures" / "F6_acc_vs_tokens.png")

    report = render_md(run_dir, prereg, summary_rows, prim_traj, mem_rows, tests, figs, synthetic, recs)
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "reports" / "findings.md").write_text(report)
    status = RunStatus.COMPLETE if read_json(run_dir / "status.json").get("status") == "COMPLETE" else RunStatus.PARTIAL
    write_status(run_dir / "metrics", "analyze", status,
                 {"synthetic": synthetic, "n_predictions": len(recs), "figures": figs})
    return {"status": status.value, "synthetic": synthetic, "tests": tests}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _mean_over_orders(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], r["episode"])].append(r)
    out = []
    for (m, t), rs in sorted(g.items()):
        out.append({"method": m, "episode": t,
                    **{k: _mean([r.get(k) for r in rs]) for k in
                       ("old_macro_acc", "new_macro_acc", "pending_unknown_rejection", "permanent_unknown_rejection")}})
    return out


def _mean_eff(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["method"], r["budget_id"])].append(r)
    return [{"method": m, "budget_id": b, "acc": _mean([r["acc"] for r in rs]),
             "mean_total_tokens": _mean([r["mean_total_tokens"] for r in rs])} for (m, b), rs in sorted(g.items())]


def render_md(run_dir, prereg, summary, traj, mem, tests, figs, synthetic, recs) -> str:
    L = [f"# CLOVE findings — run `{run_dir.name}`", ""]
    if synthetic:
        L += ["> **SYNTHETIC — NOT RESULTS.** These predictions came from the mock backend and "
              "exist only to validate the pipeline. Do not report any number below.", ""]
    L += [f"- pre-registration hash: `{prereg['hash']}` (frozen {prereg['frozen_at']})",
          f"- predictions: {len(recs)}; synthetic: {synthetic}",
          f"- primary: {prereg['stats']['primary']}", "",
          "## Pre-registered comparisons", ""]
    rows = []
    for k, v in tests.items():
        bt = v.get("bootstrap") or {}
        rows.append({"comparison": k, "family": v.get("family"), "estimate": bt.get("estimate"),
                     "ci_lo": bt.get("lo"), "ci_hi": bt.get("hi"), "p_signflip": (v.get("sign_flip") or {}).get("p"),
                     "p_adj": v.get("p_holm", v.get("p_bh")),
                     "noninferior": (v.get("noninferiority") or {}).get("noninferior"),
                     "clusters": bt.get("n_clusters"), "status": v.get("status", "")})
    L += [md_table(rows), "Non-significant differences are not evidence of equivalence; "
          "non-inferiority is claimed only when the lower CI bound clears the pre-registered margin.", "",
          "## Summary by order / budget / method", "", md_table(summary), "",
          "## Trajectories (primary budget, mean over orders)", "", md_table(traj), "",
          "## Memory-history contrast F_memory (fresh − persistent)", "", md_table(mem), "",
          "## Figures", ""]
    for k, v in figs.items():
        L.append(f"- {k}: {v or 'skipped (matplotlib unavailable or no data)'}")
    L += ["", "## Limitations carried from the protocol", "",
          "- Unknown = not in the admitted inventory; the pretrained VLM may still know these diseases.",
          "- SAGE filtered images with the same KB family now used as evidence (selection bias).",
          "- No gradient forgetting exists here (frozen weights); deterioration is agent-level.",
          "- A soybean pilot cannot establish cross-crop generality.", ""]
    return "\n".join(L)
