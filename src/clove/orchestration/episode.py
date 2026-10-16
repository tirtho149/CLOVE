"""clove.orchestration.episode — code-driven episode controller.

For each pre-committed class order × budget:
  t = 0..T-1:
    ledger.advance()  -> AccessScope_t (the only way access widens)
    every method:     policy.update(AuthorizedKB_t, Δ_t)   (no VLM calls)
                      snapshot(memory)                      (versioned derived state)
                      calibrate unknown gate on DEV images of admitted classes
                      evaluate EVERY fixed test image once  (old / new / unknown at t)
Default: one VLM call per prediction; a second call only under the opt-in,
pre-registered uncertainty band. Predictions are append-only, keyed, resumable.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Budget, Config
from ..contracts import (Decision, FailureCode, GTStatus, ImageRecord,
                         InferenceResult, Prediction, RunStatus)
from ..inference.backend import ImageGuard
from ..inference.decode import decode
from ..inference.prompt import build_prompt
from ..inference.token_meter import BudgetExceeded, BudgetMeter
from ..kb.inventory import AuthorizedKB, ClassLedger
from ..kb.reader import KBRegistry
from ..memory.base import snapshot
from ..memory.policy import MemoryPolicy, Query, Selection
from ..memory.prism import Prism
from ..memory.snapshot import make_policy
from ..data.splits import OrderPlan, StreamPlan
from ..retrieval.embedding import VisualScorer
from .policy import UnknownGate, scored_decision
from .tools import audit_episode
from .trace import ResponseCache, Trace, write_prompt, write_raw
from ..util import JsonlStore, atomic_write_json, stable_hash, write_status


class ComputeCeiling(RuntimeError):
    pass


@dataclass
class RunContext:
    cfg: Config
    run_dir: Path
    run_id: str
    registry: KBRegistry
    manifest_fp: str
    plan: StreamPlan
    backend: ImageGuard
    meter: BudgetMeter
    scorer: VisualScorer
    synthetic: bool = False
    max_new_calls: int | None = None
    max_usd: float | None = None
    counters: dict = field(default_factory=lambda: {"new_calls": 0, "cached": 0, "skipped": 0,
                                                     "predictions": 0, "errors": 0})


class EpisodeRunner:
    def __init__(self, ctx: RunContext, methods: list[str], orders: list[OrderPlan],
                 budgets: list[Budget], test_images: list[ImageRecord],
                 dev_images: list[ImageRecord]):
        self.ctx = ctx
        self.methods = methods
        self.orders = orders
        self.budgets = budgets
        self.dev_images = dev_images
        eval_classes = set(ctx.plan.streamed) | set(ctx.plan.held_back) | set(ctx.plan.permanent_unknown)
        self.test_images = [r for r in test_images if r.class_id in eval_classes]
        rd = ctx.run_dir
        self.preds = JsonlStore(rd / "predictions.jsonl")
        self.dev_preds = JsonlStore(rd / "dev_predictions.jsonl")
        self.trace = Trace(rd / "logs" / "tool_trace.jsonl")
        self.cache = ResponseCache(ctx.cfg.artifacts / "cache", ctx.backend.describe())
        self.backend_desc = ctx.backend.describe()
        self._vis_cache: dict[tuple[str, str], dict[str, float]] = {}
        self._dev_index: dict[str, dict] | None = None
        self._lock = threading.Lock()
        self._vis_lock = threading.Lock()
        self.workers = max(1, int(ctx.cfg.get("compute.parallel", 1) or 1))

    def _bump(self, k: str, n: int = 1) -> None:
        with self._lock:
            self.ctx.counters[k] = self.ctx.counters.get(k, 0) + n

    def _map(self, fn, items: list) -> list:
        """Order-preserving map; threads only help I/O-bound backends (claude -p, HTTP)."""
        if self.workers == 1 or len(items) <= 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(fn, items))

    # ── helpers ───────────────────────────────────────────────────────────
    def inference_config_hash(self, method: str, budget: Budget) -> str:
        c = self.ctx.cfg
        return stable_hash({"backend": self.backend_desc, "budget": budget.__dict__, "method": method,
                            "gate": c["gate"], "policy": c["policy"], "memory": c["memory"],
                            "prism": c["prism"] if method.startswith("prism") else None,
                            "retrieval": c["retrieval"], "tokenizer": self.ctx.meter.tokenizer.name,
                            "b_out_override": (c.get("budgets.method_max_new_tokens") or {}).get(method)})

    def class_text(self, akb: AuthorizedKB, cid: str) -> str:
        r = akb.record(cid)
        parts = f", affecting {', '.join(r.affected_parts)}" if r and r.affected_parts else ""
        return f"a photo of a soybean plant with {cid.replace('_', ' ')}{parts}"

    def visual(self, img: ImageRecord, akb: AuthorizedKB) -> dict[str, float]:
        key = (img.sha256, akb.scope.inventory_hash)
        with self._vis_lock:
            return self._visual_locked(img, akb, key)

    def _visual_locked(self, img: ImageRecord, akb: AuthorizedKB, key) -> dict[str, float]:
        if key not in self._vis_cache:
            texts = {c: self.class_text(akb, c) for c in sorted(akb.scope.allowed_class_ids)}
            self._vis_cache[key] = self.ctx.scorer.scores(img.path, img.sha256, texts, akb.scope)
        return self._vis_cache[key]

    def gt_status(self, ledger: ClassLedger, cid: str, t: int) -> tuple[str, int | None]:
        te = ledger.admit_episode_of(cid)
        if te is None:
            return GTStatus.PERMANENT_UNKNOWN.value, None
        if te < t:
            return GTStatus.OLD.value, te
        if te == t:
            return GTStatus.NEW.value, te
        return GTStatus.PENDING_UNKNOWN.value, te

    def _infer(self, pol: MemoryPolicy, img: ImageRecord, sel: Selection, budget: Budget
               ) -> tuple[InferenceResult, object, bool, str]:
        override = (self.ctx.cfg.get("budgets.method_max_new_tokens") or {}).get(pol.name)
        if override:
            budget = Budget(budget.budget_id, budget.kb_tokens, budget.r, int(override),
                            budget.instruction_tokens, budget.excerpt_tokens)
        prompt = build_prompt(sel)
        if prompt.style == "clove":
            self.ctx.meter.check("instructions", prompt.instruction_text, budget.instruction_tokens)
        ck = ResponseCache.key(img.sha256, prompt.hash, self.backend_desc, budget.max_new_tokens,
                               pol.name, sel.memory_state_hash)
        res = self.cache.get(ck)
        cached = res is not None
        if res is None:
            mx = self.ctx.max_new_calls
            mu = self.ctx.max_usd
            with self._lock:                      # reserve the call slot before calling
                if mx is not None and self.ctx.counters["new_calls"] >= mx:
                    raise ComputeCeiling(f"compute.max_calls={mx} reached")
                if mu is not None and self.ctx.counters.get("usd", 0.0) >= mu:
                    raise ComputeCeiling(f"compute.max_usd={mu} reached")
                self.ctx.counters["new_calls"] += 1
            self.ctx.backend.allow(img.path)
            res = self.ctx.backend.infer(img.path, prompt.system, prompt.user_text, budget.max_new_tokens)
            if res.cost_usd:
                self._bump("usd", float(res.cost_usd))
            with self._lock:
                self.cache.put(ck, res)
        else:
            self._bump("cached")
        write_prompt(self.ctx.run_dir, prompt.hash, prompt.system, prompt.user_text)
        raw = write_raw(self.ctx.run_dir, ck, res)
        return res, prompt, cached, raw

    def predict(self, pol: MemoryPolicy, akb: AuthorizedKB, ledger: ClassLedger, img: ImageRecord,
                t: int, order: OrderPlan, budget: Budget, gate: UnknownGate, split: str) -> Prediction:
        scope = akb.scope
        status, te = self.gt_status(ledger, img.class_id, t)
        known = status in (GTStatus.OLD.value, GTStatus.NEW.value)
        q = Query(img.image_id, t, scope, self.visual(img, akb),
                  oracle_class=img.class_id if (pol.name == "oracle" and known) else None)
        sel = pol.select(q, budget)
        problems = audit_episode(scope, ledger, sel, None, pol.index.claims if pol.index else {},
                                 self.ctx.manifest_fp, self.ctx.manifest_fp)
        failure = FailureCode.NONE
        calls = 1
        try:
            res, prompt, cached, raw = self._infer(pol, img, sel, budget)
        except BudgetExceeded:
            failure = FailureCode.CONTEXT_OVERFLOW
            res, prompt, cached, raw = InferenceResult("", 0, 0, 0.0, error="CONTEXT_OVERFLOW"), build_prompt(sel), False, ""
            calls = 0
        tot_in, tot_out, lat = res.input_tokens, res.output_tokens, res.latency_s
        d = decode(res, prompt)
        decision, label, s = gate.decide(d)
        pc = self.ctx.cfg["policy"]
        if (pc.get("second_call") and res.error is None and s is not None
                and pc["second_call_band"][0] <= s <= pc["second_call_band"][1] and pc["max_calls"] >= 2):
            wider = Budget(budget.budget_id, budget.kb_tokens, budget.r * 2, budget.max_new_tokens,
                           budget.instruction_tokens, budget.excerpt_tokens)
            sel = pol.select(q, wider)
            res2, prompt, cached2, raw = self._infer(pol, img, sel, wider)
            calls += 1
            tot_in += res2.input_tokens
            tot_out += res2.output_tokens
            lat += res2.latency_s
            d = decode(res2, prompt)
            decision, label, s = gate.decide(d)
            res = res2
        if label is not None:
            scope.check(label, "predicted label")
        if res.error:
            e = res.error
            failure = (FailureCode.TIMEOUT if "TIMEOUT" in e or "timed out" in e.lower() else
                       FailureCode.OOM if "OOM" in e else
                       FailureCode.CONTEXT_OVERFLOW if "CONTEXT_OVERFLOW" in e else
                       FailureCode.REFERENCE_IMAGE_VIOLATION if "REFERENCE_IMAGE_VIOLATION" in e else
                       FailureCode.OUTPUT_CAP_EXCEEDED if "OUTPUT_CAP_EXCEEDED" in e else
                       FailureCode.INVALID_IMAGE if "image" in e.lower() and "decode" in e.lower() else
                       FailureCode.UNKNOWN_ROOT_CAUSE)
            self._bump("errors")
        elif not d.ok:
            failure = FailureCode.OUTPUT_PARSE
        elif problems:
            failure = FailureCode.CITATION_MISMATCH
        elif known:
            if decision == Decision.KNOWN and label == img.class_id:
                failure = FailureCode.NONE
            elif img.class_id not in sel.candidates:
                failure = FailureCode.RETRIEVAL_MISS
            elif decision != Decision.KNOWN:
                failure = FailureCode.KNOWN_FALSE_UNKNOWN
            else:
                card = pol.state.cards.get(img.class_id)
                shown = {l.claim_id for l in sel.evidence.get(img.class_id, [])}
                lossy = card is not None and sel.dropped_claims > 0 and not set(card.claim_ids) <= shown
                failure = FailureCode.LOSSY_MEMORY if lossy else FailureCode.CLASS_COMPETITION
        elif decision == Decision.KNOWN:
            failure = FailureCode.UNKNOWN_FALSE_KNOWN
        ich = self.inference_config_hash(pol.name, budget)
        key = stable_hash([img.image_id, img.variant_id, t, pol.name, order.order_id, order.seed,
                           budget.budget_id, ich, split], 32)
        b_out_used = (self.ctx.cfg.get("budgets.method_max_new_tokens") or {}).get(pol.name) \
            or budget.max_new_tokens
        return Prediction(
            key=key, run_id=self.ctx.run_id, order_id=order.order_id, seed=order.seed, episode=t,
            method=pol.name, budget_id=budget.budget_id, image_id=img.image_id,
            variant_id=img.variant_id, shift_type=img.shift_type, severity=img.severity,
            group_id=img.group_id, source_dataset=img.source_dataset, gt_class=img.class_id,
            gt_status=status, gt_admit_episode=te, decision=scored_decision(decision), label_id=label,
            raw_answer=d.raw_answer, unknown_score=s,
            known_scores={k: round(v, 6) for k, v in d.probs.items()}, candidates=sel.candidates,
            evidence_ids=[l.claim_id for ls in sel.evidence.values() for l in ls],
            kb_tokens=sel.kb_tokens, prompt_tokens=tot_in, output_tokens=tot_out,
            image_tokens=res.image_tokens, vlm_calls=calls, latency_s=lat, failure=failure.value,
            class_set_hash=scope.inventory_hash, memory_state_hash=sel.memory_state_hash,
            prompt_hash=prompt.hash, inference_config_hash=ich, gate_threshold=gate.threshold,
            raw_response_path=raw, cached=cached, synthetic=bool(res.synthetic or self.ctx.synthetic),
            extra={"split": split, "score_source": d.score_source, "dropped_claims": sel.dropped_claims,
                   "audit_problems": problems, "unresolved": decision == Decision.UNRESOLVED,
                   "tokenizer_exact": self.ctx.meter.tokenizer.exact,
                   "cost_usd": res.cost_usd, "cost_incurred": (0.0 if cached else res.cost_usd),
                   # claude -p silently continues past CLAUDE_CODE_MAX_OUTPUT_TOKENS (several API
                   # requests summed into one result): flag any record whose output broke b_out
                   "exceeded_b_out": tot_out > int(b_out_used),
                   "backend_meta": res.meta, "error": res.error, **sel.notes})

    # ── main loop ─────────────────────────────────────────────────────────
    def dev_for(self, scope, n_per_class: int) -> list[ImageRecord]:
        out: dict[str, list[ImageRecord]] = {}
        for r in self.dev_images:
            if r.class_id in scope.allowed_class_ids and len(out.setdefault(r.class_id, [])) < n_per_class:
                out[r.class_id].append(r)
        return [r for c in sorted(out) for r in out[c]]

    def run(self) -> RunStatus:
        cfg = self.ctx.cfg
        gcfg = cfg["gate"]
        try:
            for order in self.orders:
                for budget in self.budgets:
                    self._run_stream(order, budget, gcfg)
        except ComputeCeiling as e:
            self.trace.log("compute_ceiling", detail=str(e))
            return RunStatus.PARTIAL
        return RunStatus.COMPLETE

    def _run_stream(self, order: OrderPlan, budget: Budget, gcfg: dict) -> None:
        cfg = self.ctx.cfg
        ledger = ClassLedger([list(b) for b in order.schedule])
        pols = {m: make_policy(m, cfg, self.ctx.meter, seed=order.seed) for m in self.methods}
        gates = {m: UnknownGate(gcfg["mode"], gcfg["target_known_tpr"]) for m in self.methods}
        mask = cfg.get("memory.mask_unadmitted_mentions", True)
        for t in range(len(order.schedule)):
            scope = ledger.advance()
            akb = AuthorizedKB(self.ctx.registry, scope, mask)
            new = tuple(order.schedule[t])
            self.trace.log("admit_class", order=order.order_id, episode=t, classes=list(new),
                           inventory_hash=scope.inventory_hash)
            for m, pol in pols.items():
                pol.update(akb, new, budget)
                h = snapshot(pol.state, self.ctx.run_dir / "memory" / order.order_id / budget.budget_id
                             / m / f"ep{t}.json")
                self.trace.log("compile_memory", order=order.order_id, episode=t, method=m, memory_hash=h,
                               masked_claims=dict(akb.masked_claims))
                gate = gates[m]
                # dev calls are only useful when they calibrate a threshold or feed PRISM's
                # protection prior; with argmax_only gating they are skipped for other methods
                needs_dev = gcfg["mode"] == "threshold" or isinstance(pol, Prism)
                if needs_dev and (gcfg["calibrate"] == "per_episode" or t == 0):
                    scores = []
                    errs: dict[str, list[int]] = {}
                    dev = self.dev_for(scope, int(gcfg["dev_per_class"]))
                    recs = self._map(lambda im: self._get_or_make(
                        self.dev_preds, pol, akb, ledger, im, t, order, budget,
                        UnknownGate("argmax_only"), "dev"), dev)
                    for img, p in zip(dev, recs):
                        if p["unknown_score"] is not None:
                            scores.append(p["unknown_score"])
                        ok = p["decision"] == Decision.KNOWN.value and p["label_id"] == img.class_id
                        errs.setdefault(img.class_id, []).append(0 if ok else 1)
                    tau = gate.calibrate(scores)
                    self.trace.log("gate_unknown.calibrate", order=order.order_id, episode=t, method=m,
                                   threshold=tau, n=gate.n_calibration)
                    if isinstance(pol, Prism):
                        pol.set_protect_prior({c: sum(v) / len(v) for c, v in errs.items() if v})
                self._map(lambda im: self._get_or_make(self.preds, pol, akb, ledger, im, t, order,
                                                       budget, gate, "test"), self.test_images)
                atomic_write_json(self.ctx.run_dir / "progress.json",
                                  {"order": order.order_id, "budget": budget.budget_id, "episode": t,
                                   "method": m, "counters": self.ctx.counters, "time": time.time()})

    def _get_or_make(self, store: JsonlStore, pol, akb, ledger, img, t, order, budget, gate, split) -> dict:
        ich = self.inference_config_hash(pol.name, budget)
        key = stable_hash([img.image_id, img.variant_id, t, pol.name, order.order_id, order.seed,
                           budget.budget_id, ich, split], 32)
        if key in store:
            self._bump("skipped")
            if split == "test":
                return {}                    # resume: the test path never reads back
            with self._lock:
                if self._dev_index is None or key not in self._dev_index:
                    self._dev_index = {r["key"]: r for r in store.iter()}
                return self._dev_index[key]
        p = self.predict(pol, akb, ledger, img, t, order, budget, gate, split)
        rec = p.to_json()
        if rec["key"] != key:
            raise AssertionError("prediction key drift")
        with self._lock:
            store.append(rec)
            self.ctx.counters["predictions"] += 1
        return rec
