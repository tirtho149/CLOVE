"""clove.orchestration.plan_calls — exact dry-run call count for `clove estimate`.

Builds every prompt the pilot would send (same ledger, policies, dev calibration and
test loop as `EpisodeRunner`) WITHOUT calling the model, and counts distinct
response-cache keys. Byte-identical inputs are served from the cache at run time, so
this is the number of billable calls. Requires a deterministic candidate retriever
(`retrieval.visual: none` or `precomputed`); with SigLIP only the upper bound is known.
"""
from __future__ import annotations

from collections import Counter

from ..config import Config
from ..inference.prompt import build_prompt
from ..kb.inventory import AuthorizedKB, ClassLedger
from ..memory.policy import Query
from ..memory.snapshot import make_policy
from ..retrieval.embedding import NoScorer, PrecomputedScorer


def exact_unique_calls(cfg: Config, registry, man, plan, methods: list[str], orders, budget, meter) -> dict:
    vis = cfg.get("retrieval.visual")
    if vis == "siglip":
        raise RuntimeError("exact count needs a deterministic retriever (visual: none | precomputed)")
    scorer = PrecomputedScorer(cfg.get("retrieval.precomputed_scores")) if vis == "precomputed" else NoScorer()
    eval_classes = set(plan.streamed) | set(plan.held_back) | set(plan.permanent_unknown)
    test = [r for r in man.images if r.split == "test" and r.class_id in eval_classes]
    dev_all = [r for r in man.images if r.split == "dev"]
    n_dev = int(cfg.get("gate.dev_per_class"))
    overrides = cfg.get("budgets.method_max_new_tokens") or {}
    mask = cfg.get("memory.mask_unadmitted_mentions", True)
    keys: set = set()
    total = 0
    by_method: Counter = Counter()
    by_episode: Counter = Counter()          # distinct (billable) calls per episode t
    by_order_episode: Counter = Counter()
    planned_by_episode: Counter = Counter()
    kb_tokens: Counter = Counter()
    for order in orders:
        ledger = ClassLedger([list(b) for b in order.schedule])
        pols = {m: make_policy(m, cfg, meter, seed=order.seed) for m in methods}
        for t in range(len(order.schedule)):
            scope = ledger.advance()
            akb = AuthorizedKB(registry, scope, mask)
            for m, pol in pols.items():
                pol.update(akb, tuple(order.schedule[t]), budget)
                per: dict[str, int] = {}
                dev = []
                for r in dev_all:
                    if r.class_id in scope.allowed_class_ids and per.get(r.class_id, 0) < n_dev:
                        dev.append(r)
                        per[r.class_id] = per.get(r.class_id, 0) + 1
                from ..memory.prism import Prism
                needs_dev = cfg.get("gate.mode") == "threshold" or isinstance(pol, Prism)
                calib = cfg.get("gate.calibrate") == "per_episode" or t == 0
                for img in (dev if needs_dev and calib else []) + test:
                    texts = {}
                    vs = scorer.scores(img.path, img.sha256, texts, scope) if vis == "precomputed" else {}
                    sel = pol.select(Query(img.image_id, t, scope, vs), budget)
                    pr = build_prompt(sel)
                    k = (img.sha256, pr.hash, overrides.get(m) or budget.max_new_tokens, m, sel.memory_state_hash)
                    total += 1
                    planned_by_episode[t] += 1
                    if k not in keys:
                        keys.add(k)
                        by_method[m] += 1
                        by_episode[t] += 1
                        by_order_episode[f"{order.order_id}|t{t}"] += 1
                        kb_tokens[m] += sel.kb_tokens
    return {"planned_predictions": total, "unique_calls": len(keys),
            "unique_calls_by_method": dict(by_method),
            "unique_calls_by_episode": {f"t{t}": n for t, n in sorted(by_episode.items())},
            "planned_by_episode": {f"t{t}": n for t, n in sorted(planned_by_episode.items())},
            "unique_calls_by_order_episode": dict(sorted(by_order_episode.items())),
            "mean_kb_tokens_by_method": {m: kb_tokens[m] / max(1, by_method[m]) for m in by_method},
            "n_test": len(test)}
