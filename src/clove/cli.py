"""clove.cli — one command per protocol gate. Every command writes `status.json`
(COMPLETE / PARTIAL / BLOCKED), exits nonzero on failure, and never substitutes
fabricated data for missing inputs.

  python -m clove.cli audit        --config configs/pilot.yaml
  python -m clove.cli manifest     --config configs/pilot.yaml
  python -m clove.cli test-leakage --config configs/pilot.yaml
  python -m clove.cli smoke        --config configs/pilot.yaml [--synthetic]
  python -m clove.cli estimate     --config configs/pilot.yaml
  python -m clove.cli run          --config configs/pilot.yaml --methods names,sage0,append,fresh,prism
  python -m clove.cli analyze      --run-dir artifacts/runs/<RUN_ID>
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path

from .config import Config
from .contracts import FutureClassLeak, RunStatus
from .util import atomic_write_json, read_json, stable_hash, write_status


def _cfg(a) -> Config:
    over = json.loads(a.override) if getattr(a, "override", None) else None
    return Config.load(a.config, over)


def _resume(a, cmd: str) -> str:
    return " ".join(["python -m clove.cli", cmd] + [shlex.quote(x) for x in sys.argv[2:]])


# ── audit ───────────────────────────────────────────────────────────────────
def cmd_audit(a) -> int:
    from .data.audit import audit, render_audit_md
    cfg = _cfg(a)
    out = cfg.artifacts / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    try:
        rep = audit(cfg)
    except FileNotFoundError as e:
        write_status(out, "audit", RunStatus.BLOCKED, {"error": str(e)}, _resume(a, "audit"))
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 2
    rep.pop("_objects")
    atomic_write_json(out / "audit.json", rep)
    md = render_audit_md(rep)
    rp = cfg.artifacts / "reports" / "audit.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(md)
    blockers = []
    if not rep["files"]["wacv_manifest"]["exists"] and not rep["files"]["prepared_dataset"]["exists"]:
        blockers.append("no image manifest")
    if len(rep["stream"]["admissible"]) < 4:
        blockers.append(f"only {len(rep['stream']['admissible'])} admissible classes")
    st = RunStatus.BLOCKED if blockers else RunStatus.COMPLETE
    write_status(out, "audit", st, {"blockers": blockers, "report": str(rp),
                                    "admissible": len(rep["stream"]["admissible"]),
                                    "test_images": sum(rep["manifest_stats"]["test_per_class"].values())},
                 _resume(a, "audit") if blockers else None)
    print(f"audit {st.value}: {rp}")
    return 0 if st == RunStatus.COMPLETE else 2


# ── manifest ────────────────────────────────────────────────────────────────
def cmd_manifest(a) -> int:
    from .resources import build_canonical_manifest, build_plan, freeze, load_registry
    cfg = _cfg(a)
    namer, reg = load_registry(cfg)
    man = build_canonical_manifest(cfg, namer)
    plan, reasons = build_plan(cfg, man, reg)
    try:
        freeze(cfg, man, plan, reasons)
    except RuntimeError as e:
        write_status(cfg.artifacts / "manifests", "manifest", RunStatus.BLOCKED, {"error": str(e)})
        print(f"BLOCKED: {e}", file=sys.stderr)
        return 2
    write_status(cfg.artifacts / "manifests", "manifest", RunStatus.COMPLETE,
                 {"manifest_fingerprint": man.fingerprint, "plan_fingerprint": plan.fingerprint,
                  "kb_fingerprint": reg.fingerprint, "orders": [o.order_id for o in plan.orders],
                  "sizes": plan.sizes})
    print(f"manifest frozen: {man.fingerprint}  orders: {[o.order_id for o in plan.orders]}  sizes {plan.sizes}")
    return 0


# ── leakage ─────────────────────────────────────────────────────────────────
def leakage_checks(cfg: Config) -> dict:
    from .contracts import AccessScope
    from .data.leakage import cross_split_leaks, label_conflicts
    from .inference.token_meter import BudgetMeter, RegexTokenizer
    from .kb.inventory import AuthorizedKB, ClassLedger, inventory_hash
    from .memory.policy import Query
    from .memory.snapshot import make_policy
    from .resources import load_manifest, load_plan, load_registry
    namer, reg = load_registry(cfg)
    man, plan = load_manifest(cfg), load_plan(cfg)
    res: dict = {"checks": {}, "violations": []}

    def check(name: str, ok: bool, detail=None):
        res["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            res["violations"].append(name)

    leaks = cross_split_leaks(man.images)
    check("no_dev_test_group_overlap", not leaks, leaks[:20])
    check("no_label_conflicting_groups", not label_conflicts(man.images))
    ids = [r.image_id for r in man.images if r.split == "test"]
    check("unique_test_ids", len(ids) == len(set(ids)))
    for o in plan.orders:
        flat = [c for b in o.schedule for c in b]
        check(f"{o.order_id}:disjoint_batches", len(flat) == len(set(flat)))
        check(f"{o.order_id}:only_streamed_classes", set(flat) == set(plan.streamed))
        check(f"{o.order_id}:no_unknown_pool_in_schedule",
              not (set(flat) & (set(plan.held_back) | set(plan.permanent_unknown))))
    # hidden-class invariant with a deliberately hidden class
    hidden = sorted(plan.streamed)[-1]
    scope = AccessScope(0, frozenset(c for c in plan.streamed if c != hidden),
                        inventory_hash([c for c in plan.streamed if c != hidden]))
    try:
        AuthorizedKB(reg, scope).claims(hidden)
        check("hidden_class_kb_request_rejected", False)
    except FutureClassLeak:
        check("hidden_class_kb_request_rejected", True)
    # dry selections for every order/episode/method: evidence only from admitted classes
    meter = BudgetMeter(RegexTokenizer())
    budget = cfg.budget()
    methods = sorted(set(cfg["methods"]) | {"append", "fresh", "prism"})
    bad = []
    for o in plan.orders:
        ledger = ClassLedger([list(b) for b in o.schedule])
        pols = {m: make_policy(m, cfg, meter, o.seed) for m in methods}
        for t in range(len(o.schedule)):
            sc = ledger.advance()
            akb = AuthorizedKB(reg, sc, cfg.get("memory.mask_unadmitted_mentions", True))
            for m, p in pols.items():
                p.update(akb, tuple(o.schedule[t]), budget)
                for cid, card in p.state.cards.items():
                    if cid not in sc.allowed_class_ids:
                        bad.append(f"{o.order_id}:t{t}:{m}:card:{cid}")
                sel = p.select(Query("leakcheck", t, sc, {}), budget)
                for cid, lines in sel.evidence.items():
                    for ln in lines:
                        cl = reg._get_claims(ln.class_id)
                        c = next((x for x in cl if x.claim_id == ln.claim_id), None)
                        if c is None or c.class_id not in sc.allowed_class_ids or \
                                any(mm not in sc.allowed_class_ids for mm in c.mentions):
                            bad.append(f"{o.order_id}:t{t}:{m}:{ln.claim_id}")
                    if cid not in sc.allowed_class_ids:
                        bad.append(f"{o.order_id}:t{t}:{m}:option:{cid}")
                if sel.kb_tokens > budget.kb_tokens and m != "sage0":
                    bad.append(f"{o.order_id}:t{t}:{m}:kb_cap")
    check("dry_selection_scope_and_caps", not bad, bad[:20])
    res["manifest_fingerprint"] = man.fingerprint
    res["plan_fingerprint"] = plan.fingerprint
    return res


def cmd_test_leakage(a) -> int:
    cfg = _cfg(a)
    try:
        res = leakage_checks(cfg)
    except FileNotFoundError as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        write_status(cfg.artifacts / "manifests", "test-leakage", RunStatus.BLOCKED, {"error": str(e)})
        return 2
    atomic_write_json(cfg.artifacts / "reports" / "leakage.json", res)
    ok = not res["violations"]
    write_status(cfg.artifacts / "reports", "test-leakage", RunStatus.COMPLETE if ok else RunStatus.BLOCKED,
                 {"violations": res["violations"]})
    print("leakage checks:", "PASS" if ok else f"FAIL {res['violations']}")
    return 0 if ok else 3


# ── shared run construction ─────────────────────────────────────────────────
def build_context(cfg: Config, run_dir: Path, synthetic: bool, allow_approx: bool):
    from .inference.backend import ImageGuard, make_backend
    from .inference.token_meter import BudgetMeter, RegexTokenizer, load_tokenizer
    from .orchestration.episode import RunContext
    from .resources import load_manifest, load_plan, load_registry
    from .retrieval.embedding import NoScorer, make_scorer
    namer, reg = load_registry(cfg)
    man, plan = load_manifest(cfg), load_plan(cfg)
    if synthetic:
        truth = {r.path: r.class_id for r in man.images}      # used ONLY by the mock
        backend = make_backend({**cfg["model"], "backend": "mock"}, truth)
        tok = RegexTokenizer()
        scorer = NoScorer() if cfg.get("retrieval.visual") == "siglip" else make_scorer(cfg["retrieval"])
    else:
        if cfg.get("model.backend") == "mock":
            raise RuntimeError("model.backend=mock is only allowed with --synthetic")
        backend = make_backend(cfg["model"])
        if hasattr(backend, "health"):
            ok, msg = backend.health()
            if not ok:
                raise ConnectionError(f"VLM backend not reachable: {msg}")
        tok = load_tokenizer(cfg["model"], allow_approx=allow_approx)
        scorer = make_scorer(cfg["retrieval"])
    ctx = RunContext(cfg=cfg, run_dir=run_dir, run_id=run_dir.name, registry=reg,
                     manifest_fp=man.fingerprint, plan=plan, backend=ImageGuard(backend),
                     meter=BudgetMeter(tok), scorer=scorer, synthetic=synthetic,
                     max_new_calls=int(cfg.get("compute.max_calls")),
                     max_usd=(float(cfg.get("compute.max_usd")) if cfg.get("compute.max_usd") is not None else None))
    return ctx, man, plan


def preregister(cfg: Config, run_dir: Path, methods: list[str], plan_fp: str, man_fp: str) -> dict:
    body = {"stats": cfg["stats"], "methods": methods, "primary_budget": cfg["budgets"]["primary"],
            "gate": cfg["gate"], "policy": cfg["policy"], "prism": cfg["prism"],
            "method_max_new_tokens": cfg.get("budgets.method_max_new_tokens") or {},
            "plan_fingerprint": plan_fp, "manifest_fingerprint": man_fp,
            "prevalence": cfg.get("shifts.prevalence")}
    h = stable_hash(body)
    p = run_dir / "preregistration.json"
    if p.exists():
        old = read_json(p)
        if old["hash"] != h:
            raise RuntimeError(f"pre-registration changed (frozen {old['hash']} vs now {h}); "
                               "start a new run id instead of editing a registered analysis")
        return old
    rec = {**body, "hash": h, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    atomic_write_json(p, rec)
    return rec


def _run(cfg: Config, a, methods: list[str], orders, budgets, test, dev, run_dir: Path,
         synthetic: bool, cmd: str) -> int:
    from .orchestration.episode import EpisodeRunner
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        ctx, man, plan = build_context(cfg, run_dir, synthetic, getattr(a, "allow_approx_tokenizer", False))
    except (ConnectionError, RuntimeError, OSError) as e:
        write_status(run_dir, cmd, RunStatus.BLOCKED, {"error": str(e)}, _resume(a, cmd))
        print(f"BLOCKED: {e}\nresume: {_resume(a, cmd)}", file=sys.stderr)
        return 2
    preregister(cfg, run_dir, methods, plan.fingerprint, man.fingerprint)
    atomic_write_json(run_dir / "run_config.json", {"config": cfg.data, "config_hash": cfg.hash(),
                                                    "backend": ctx.backend.describe(),
                                                    "tokenizer": ctx.meter.tokenizer.name,
                                                    "tokenizer_exact": ctx.meter.tokenizer.exact,
                                                    "synthetic": synthetic, "argv": sys.argv})
    runner = EpisodeRunner(ctx, methods, orders, budgets, test, dev)
    st = runner.run()
    b = ctx.backend.backend
    if hasattr(b, "assert_frozen"):
        b.assert_frozen()
    write_status(run_dir, cmd, st, {"counters": ctx.counters, "synthetic": synthetic,
                                    "n_orders": len(orders), "methods": methods},
                 None if st == RunStatus.COMPLETE else _resume(a, cmd))
    print(f"{cmd} {st.value}: {run_dir}  {ctx.counters}")
    return 0 if st == RunStatus.COMPLETE else 1


# ── smoke (Gate 3 micro-run) ────────────────────────────────────────────────
def cmd_smoke(a) -> int:
    from .data.splits import OrderPlan
    from .resources import load_manifest, load_plan
    cfg = _cfg(a)
    man, plan = load_manifest(cfg), load_plan(cfg)
    sm = cfg["smoke"]
    base = plan.orders[0]
    flat = [c for b in base.schedule for c in b]
    k = max(2, min(4, int(sm["classes"])))
    sched = [flat[:k - 1], flat[k - 1:k]]
    order = OrderPlan(f"SMOKE_{base.order_id}", "smoke", base.seed, sched)
    unk = (plan.permanent_unknown + plan.held_back + flat[k:])[:1]
    keep = set(flat[:k]) | set(unk)
    test, dev, nt, nd = [], [], {}, {}
    for r in man.images:
        if r.class_id not in keep:
            continue
        if r.split == "test" and nt.get(r.class_id, 0) < int(sm["images_per_class"]):
            test.append(r); nt[r.class_id] = nt.get(r.class_id, 0) + 1
        if r.split == "dev" and nd.get(r.class_id, 0) < int(sm["dev_per_class"]):
            dev.append(r); nd[r.class_id] = nd.get(r.class_id, 0) + 1
    plan.streamed, plan.held_back, plan.permanent_unknown = flat[:k], [], unk
    root = cfg.artifacts / ("synthetic_runs" if a.synthetic else "runs")
    run_dir = root / f"smoke_{cfg.hash()[:8]}"
    methods = ["sage0", "append", "fresh"]
    cfg.data["methods"] = methods
    return _run(cfg, a, methods, [order], [cfg.budget()], test, dev, run_dir, a.synthetic, "smoke")


# ── estimate ────────────────────────────────────────────────────────────────
def claude_cost_model(cfg: Config) -> dict | None:
    """Fit $/token (input, cache write, cache read, output) from every real smoke call
    with the same Claude model, and per prompt family (CLOVE vs SAGE k=0) the mean
    prompt/output tokens and observed cache-read share."""
    import numpy as np
    model = cfg.get("model.claude.model")
    rows = []
    for p in (cfg.artifacts / "runs").glob("smoke_*/*predictions.jsonl"):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            mu = ((r.get("extra") or {}).get("backend_meta") or {}).get("model_usage") or {}
            if r.get("cached") or model not in mu:
                continue
            u = mu[model]
            rows.append((u["inputTokens"], u["cacheCreationInputTokens"], u["cacheReadInputTokens"],
                         u["outputTokens"], u["costUSD"], "sage0" if r["method"] == "sage0" else "clove"))
    if len(rows) < 8:
        return None
    X = np.array([r[:4] for r in rows], float)
    price = np.linalg.lstsq(X, np.array([r[4] for r in rows]), rcond=None)[0]
    fams = {}
    for f in ("clove", "sage0"):
        s = [r for r in rows if r[5] == f]
        if not s:
            continue
        tot_in = np.mean([r[0] + r[1] + r[2] for r in s])
        out = np.mean([r[3] for r in s])
        repeat = [r for r in s if r[2] > 0]         # calls that found a warm prompt cache
        share = float(np.mean([r[2] / (r[0] + r[1] + r[2]) for r in repeat])) if repeat else 0.0
        fams[f] = {"n": len(s), "mean_prompt_tokens": float(tot_in), "mean_output_tokens": float(out),
                   "warm_cache_read_share": share,
                   "usd_per_call_no_cache": float(tot_in * price[1] + out * price[3]),
                   "usd_per_call_observed_cache_mix": float(tot_in * (share * price[2] + (1 - share) * price[1])
                                                            + out * price[3])}
    return {"usd_per_mtok": {"input": price[0] * 1e6, "cache_write": price[1] * 1e6,
                             "cache_read": price[2] * 1e6, "output": price[3] * 1e6},
            "n_calls": len(rows), "families": fams,
            "assumption": "every repeat send of an image finds a warm prompt cache (as in the smoke); "
                          "first sight of each image/prefix is a cache write"}


def cmd_estimate(a) -> int:
    from .resources import load_manifest, load_plan
    cfg = _cfg(a)
    man, plan = load_manifest(cfg), load_plan(cfg)
    eval_classes = set(plan.streamed) | set(plan.held_back) | set(plan.permanent_unknown)
    N = sum(1 for r in man.images if r.split == "test" and r.class_id in eval_classes)
    D = min(int(cfg.get("gate.dev_per_class")), 8) * len(plan.streamed)
    E = len(plan.sizes)
    M_ = len(cfg["methods"])
    O_ = len(plan.orders)
    B = len(cfg["budgets"]["grid"]) if cfg.get("budgets.sweep_enabled") else 1
    V = int(cfg.get("policy.max_calls")) if cfg.get("policy.second_call") else 1
    calls_hi = (N + D) * E * M_ * O_ * B * V
    tbar = a.seconds_per_call
    src = "--seconds-per-call"
    backend = cfg.get("model.backend")
    times, costs = [], []
    cmodel = cfg.get("model.claude.model") if backend == "claude_headless" else None
    pattern = "smoke_*/predictions.jsonl" if cmodel else f"smoke_{cfg.hash()[:8]}/predictions.jsonl"
    for p in (cfg.artifacts / "runs").glob(pattern):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("cached") or r.get("synthetic") or r["latency_s"] <= 0:
                continue
            if cmodel and cmodel not in (((r.get("extra") or {}).get("backend_meta") or {}).get("model_usage") or {}):
                continue                      # timings from a different model don't apply
            times.append(r["latency_s"])
            c = (r.get("extra") or {}).get("cost_usd")
            if c is not None:
                costs.append(float(c))
    if tbar is None and times:
        tbar = sum(times) / len(times)
        src = f"measured mean of {len(times)} real smoke calls"
    eta = float(a.utilization)
    rep = {"N_test": N, "dev_calibration_per_episode": D, "E": E, "M": M_, "O": O_, "B": B, "V": V,
           "calls_upper_bound": calls_hi, "seconds_per_call": tbar, "seconds_per_call_source": src,
           "utilization_eta": eta,
           "note": "upper bound: assumes no cache hits; fresh/names/sage0 with identical inputs reuse cache"}
    if cfg.get("retrieval.visual") != "siglip":
        from .inference.token_meter import BudgetMeter, RegexTokenizer
        from .orchestration.plan_calls import exact_unique_calls
        from .resources import load_registry
        _, reg = load_registry(cfg)
        sel_orders = [o for o in plan.orders if not a.orders or o.order_id in a.orders.split(",")]
        rep["orders"] = [o.order_id for o in sel_orders]
        ex = exact_unique_calls(cfg, reg, man, plan, list(cfg["methods"]), sel_orders, cfg.budget(),
                                BudgetMeter(RegexTokenizer()))
        rep["exact"] = ex
        rep["calls_upper_bound_naive"] = calls_hi
        calls_hi = ex["unique_calls"] * B * V
        rep["calls_upper_bound"] = calls_hi
        rep["note"] = ("exact dry-run count of distinct prompts (cache dedupes byte-identical inputs); "
                       "x V if second calls are enabled")
    if backend == "claude_headless":
        par = max(1, int(cfg.get("compute.parallel", 1)))
        by_m: dict[str, list[float]] = {}
        for p in (cfg.artifacts / "runs").glob(f"smoke_{cfg.hash()[:8]}/predictions.jsonl"):
            for line in p.read_text().splitlines():
                r = json.loads(line)
                c = (r.get("extra") or {}).get("cost_usd")
                if not r.get("cached") and c is not None:
                    by_m.setdefault(r["method"], []).append(float(c))
        cbar = a.usd_per_call if a.usd_per_call is not None else (sum(costs) / len(costs) if costs else None)
        if "exact" in rep and a.usd_per_call is None:
            cm = claude_cost_model(cfg)
            if cm:
                rep["cost_model"] = cm
                exp = worst = 0.0
                for m, n in rep["exact"]["unique_calls_by_method"].items():
                    fam = cm["families"].get("sage0" if m == "sage0" else "clove")
                    if fam is None:
                        continue
                    exp += n * fam["usd_per_call_observed_cache_mix"]
                    worst += n * fam["usd_per_call_no_cache"]
                rep["usd_expected"] = exp * B * V
                rep["usd_worst_no_prompt_cache"] = worst * B * V
                cbar = exp / max(1, rep["exact"]["unique_calls"])
        rep.update({"backend": backend, "usd_per_call": cbar, "parallel_workers": par,
                    "usd_source": "--usd-per-call" if a.usd_per_call is not None else f"{len(costs)} real smoke calls"})
        if cbar is None or tbar is None:
            rep["status"] = "BLOCKED: no cost/timing — run `smoke` with the claude backend first"
            st = RunStatus.BLOCKED
        else:
            rep["usd_upper_bound"] = calls_hi * cbar
            rep["wall_clock_hours_upper_bound"] = calls_hi * tbar / 3600 / par
            rep["approved_budget_usd"] = cfg.get("compute.max_usd")
            rep["within_budget"] = rep["usd_upper_bound"] <= float(cfg.get("compute.max_usd"))
            st = RunStatus.COMPLETE if rep["within_budget"] else RunStatus.BLOCKED
    elif tbar is None:
        rep["status"] = "BLOCKED: no timing — run `smoke` against the real backend or pass --seconds-per-call"
        st = RunStatus.BLOCKED
    else:
        par = max(1, int(cfg.get("compute.parallel", 1)))
        n_gpus = max(1, int(cfg.get("compute.n_gpus", 1)))
        # tbar is per-request latency measured under the same concurrency, so with `par`
        # requests in flight wall-clock ~ calls * tbar / par; GPU-hours = wall * n_gpus.
        wall_h = calls_hi * tbar / par / 3600 / eta
        gpu_h = wall_h * n_gpus
        rep.update({"parallel_requests": par, "n_gpus": n_gpus, "wall_clock_hours_upper_bound": wall_h})
        rep["gpu_hours_upper_bound"] = gpu_h
        rep["approved_budget_gpu_hours"] = cfg.get("compute.max_gpu_hours")
        rep["within_budget"] = gpu_h <= float(cfg.get("compute.max_gpu_hours"))
        st = RunStatus.COMPLETE if rep["within_budget"] else RunStatus.BLOCKED
    p = cfg.artifacts / "reports" / "compute_estimate.json"
    atomic_write_json(p, rep)
    write_status(cfg.artifacts / "reports", "estimate", st, rep)
    print(json.dumps(rep, indent=2))
    return 0 if st == RunStatus.COMPLETE else 2


# ── run (Gate 4) ────────────────────────────────────────────────────────────
def cmd_run(a) -> int:
    from .data.transforms import make_variants
    from .resources import load_manifest, load_plan
    cfg = _cfg(a)
    methods = a.methods.split(",") if a.methods else list(cfg["methods"])
    cfg.data["methods"] = methods
    cfg.validate()
    lk = cfg.artifacts / "reports" / "leakage.json"
    if not lk.exists() or read_json(lk)["violations"]:
        print("BLOCKED: run `test-leakage` first (and fix violations)", file=sys.stderr)
        return 2
    est = cfg.artifacts / "reports" / "compute_estimate.json"
    if not a.synthetic and not a.skip_estimate and (not est.exists() or not read_json(est).get("within_budget")):
        print("BLOCKED: `estimate` missing or over the approved GPU-hour budget", file=sys.stderr)
        return 2
    man, plan = load_manifest(cfg), load_plan(cfg)
    orders = [o for o in plan.orders if not a.orders or o.order_id in a.orders.split(",")]
    budgets = cfg.budgets() if cfg.get("budgets.sweep_enabled") else [cfg.budget()]
    test = man.by_split("test")
    if "covariate" in (cfg.get("shifts.enabled") or []):
        test = test + make_variants(test, cfg.get("shifts.covariate"), str(cfg.artifacts / "variants"))
    dev = man.by_split("dev")
    if a.limit_images:
        per: dict[str, int] = {}
        lim = []
        for r in test:
            if per.get(r.class_id, 0) < a.limit_images:
                lim.append(r); per[r.class_id] = per.get(r.class_id, 0) + 1
        test = lim
    run_id = a.run_id or f"{cfg['run_name']}_{stable_hash([cfg.hash(), methods, [o.order_id for o in orders]])[:10]}"
    root = cfg.artifacts / ("synthetic_runs" if a.synthetic else "runs")
    return _run(cfg, a, methods, orders, budgets, test, dev, root / run_id, a.synthetic, "run")


def cmd_analyze(a) -> int:
    from .reports.render import analyze
    rd = Path(a.run_dir)
    if not (rd / "predictions.jsonl").exists():
        print(f"BLOCKED: no predictions in {rd}", file=sys.stderr)
        return 2
    res = analyze(rd)
    print(f"analyze {res['status']}{' (SYNTHETIC — not results)' if res.get('synthetic') else ''}: "
          f"{rd / 'reports' / 'findings.md'}")
    return 0 if res["status"] in ("COMPLETE", "PARTIAL") else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clove")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, cfg=True):
        p = sub.add_parser(name)
        if cfg:
            p.add_argument("--config", required=True)
            p.add_argument("--override", help="JSON dict merged over the config (logged in run_config)")
        p.set_defaults(fn=fn)
        return p

    add("audit", cmd_audit)
    add("manifest", cmd_manifest)
    add("test-leakage", cmd_test_leakage)
    p = add("smoke", cmd_smoke)
    p.add_argument("--synthetic", action="store_true", help="mock backend; pipeline validation only")
    p.add_argument("--allow-approx-tokenizer", action="store_true")
    p = add("estimate", cmd_estimate)
    p.add_argument("--seconds-per-call", type=float)
    p.add_argument("--utilization", type=float, default=0.8)
    p.add_argument("--usd-per-call", type=float, help="override measured cost (claude_headless)")
    p.add_argument("--orders", help="comma-separated order ids to estimate (default: all)")
    p = add("run", cmd_run)
    p.add_argument("--methods")
    p.add_argument("--orders", help="comma-separated order ids (default: all pre-committed)")
    p.add_argument("--run-id")
    p.add_argument("--limit-images", type=int, help="max test images per class (staged expansion)")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--skip-estimate", action="store_true")
    p.add_argument("--allow-approx-tokenizer", action="store_true")
    p = add("analyze", cmd_analyze, cfg=False)
    p.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
