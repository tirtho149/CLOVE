"""clove.data.splits — eligibility, feasible schedule, and pre-committed class orders.

All orders are generated from a seed and frozen to ``artifacts/orders/orders.json``
(with a hash) by ``clove manifest`` BEFORE any inference runs.
"""
from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..kb.reader import KBRegistry
from ..retrieval.text import TfIdf
from ..util import stable_hash


@dataclass
class Eligibility:
    admissible: list[str]            # images + KB -> can enter the class stream
    permanent_unknown: list[str]     # images but no usable KB (or held out) -> never admitted
    kb_only: list[str]               # KB but too few test images -> not evaluated
    reasons: dict[str, str] = field(default_factory=dict)


def eligibility(test_counts: dict[str, int], reg: KBRegistry, *, min_test_images: int,
                min_claims: int, hold_out: list[str], exclude: list[str]) -> Eligibility:
    adm, unk, kbo, why = [], [], [], {}
    for cid in sorted(set(test_counts) | set(reg.class_ids)):
        n_img, n_kb = test_counts.get(cid, 0), reg.n_claims(cid)
        if cid in exclude:
            why[cid] = "excluded_by_config"
        elif n_img < min_test_images:
            if n_kb:
                kbo.append(cid)
            why[cid] = f"test_images={n_img}<{min_test_images}"
        elif cid in hold_out:
            unk.append(cid)
            why[cid] = "held_out_by_config"
        elif n_kb < min_claims:
            unk.append(cid)
            why[cid] = f"kb_claims={n_kb}<{min_claims} (no knowledge to admit)"
        else:
            adm.append(cid)
            why[cid] = "admissible"
    return Eligibility(adm, unk, kbo, why)


def derive_schedule(n: int, initial: int, batch: int, stages: int, min_initial: int = 4,
                    min_batch: int = 2) -> tuple[list[int], int, list[str]]:
    """Returns (stage sizes, n_held_back, departures). Prefers the configured schedule,
    holds surplus classes back as never-admitted unknowns, and otherwise shrinks
    initial -> batch -> stages, logging every departure."""
    notes: list[str] = []
    for st in range(stages, 1, -1):
        for b in range(batch, min_batch - 1, -1):
            need_after = b * (st - 1)
            init = min(initial, n - need_after)
            if init >= min_initial:
                sizes = [init] + [b] * (st - 1)
                held = n - sum(sizes)
                if st != stages:
                    notes.append(f"stages reduced {stages}->{st}")
                if b != batch:
                    notes.append(f"batch reduced {batch}->{b}")
                if init != initial:
                    notes.append(f"initial reduced {initial}->{init}")
                if held:
                    notes.append(f"{held} admissible classes held back as never-admitted unknowns")
                return sizes, held, notes
    raise ValueError(f"only {n} admissible classes: cannot form >=2 episodes "
                     f"(min_initial={min_initial}, min_batch={min_batch})")


def _stratified(classes: list[str], strata: dict[str, str], rng: random.Random) -> list[str]:
    groups: dict[str, list[str]] = {}
    for c in sorted(classes):
        groups.setdefault(strata.get(c) or "unknown_type", []).append(c)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups, key=lambda k: (-len(groups[k]), k))
    out: list[str] = []
    while any(groups.values()):
        for k in keys:
            if groups[k]:
                out.append(groups[k].pop())
    return out


def _cut(seq: list[str], sizes: list[int]) -> list[list[str]]:
    out, i = [], 0
    for s in sizes:
        out.append(sorted(seq[i:i + s]))
        i += s
    return out


def hard_pairs(admissible: list[str], reg: KBRegistry, source: str,
               csv_path: str | None = None, top: int = 8) -> tuple[list[tuple[str, str, float]], str]:
    """Documented confusable pairs. `registry_look_alikes` uses SAGE's source-cited
    look-alike field; `csv` reads an independent list; `kb_similarity_proxy` is the
    transparent fallback (TF-IDF over KB text) and is labelled as a proxy."""
    adm = set(admissible)
    pairs: dict[tuple[str, str], float] = {}
    label = source
    if source == "csv" and csv_path and Path(csv_path).exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                a, b = row["class_a"], row["class_b"]
                if a in adm and b in adm and a != b:
                    pairs[tuple(sorted((a, b)))] = float(row.get("weight", 1) or 1)  # type: ignore[index]
    elif source == "registry_look_alikes":
        for a in admissible:
            r = reg._get_record(a)
            for b in (r.look_alikes if r else ()):
                if b in adm and b != a:
                    k = tuple(sorted((a, b)))
                    pairs[k] = pairs.get(k, 0) + 1.0  # type: ignore[index]
    if not pairs:
        label = "kb_similarity_proxy"
        docs = {c: " ".join(x.text for x in reg._get_claims(c)) for c in admissible}
        tf = TfIdf(docs)
        sims = sorted(((tf.sim(a, b), a, b) for i, a in enumerate(admissible)
                       for b in admissible[i + 1:]), reverse=True)
        for s, a, b in sims[:top]:
            pairs[(a, b)] = s
    ranked = sorted(((a, b, w) for (a, b), w in pairs.items()), key=lambda x: (-x[2], x[0], x[1]))
    return ranked, label


def _confusable_order(sizes: list[int], pairs: list[tuple[str, str, float]], classes: list[str],
                      strata: dict[str, str], rng: random.Random, reverse: bool
                      ) -> tuple[list[list[str]], list[tuple[str, str]]]:
    used: set[str] = set()
    chosen: list[tuple[str, str]] = []
    for a, b, _ in pairs:              # disjoint pairs, strongest first
        if a in used or b in used:
            continue
        early, late = (b, a) if reverse else (a, b)
        chosen.append((early, late))
        used |= {a, b}
    later_slots = sum(sizes[1:])
    chosen = chosen[:min(sizes[0], later_slots)]
    stages: list[list[str]] = [[] for _ in sizes]
    for early, _ in chosen:
        stages[0].append(early)
    for i, (_, late) in enumerate(chosen):
        t = 1 + i % (len(sizes) - 1)
        while len(stages[t]) >= sizes[t]:
            t = 1 + (t % (len(sizes) - 1))
        stages[t].append(late)
    placed = {c for s in stages for c in s}
    rest = _stratified([c for c in classes if c not in placed], strata, rng)
    for t, size in enumerate(sizes):
        while len(stages[t]) < size:
            stages[t].append(rest.pop(0))
    return [sorted(s) for s in stages], chosen


@dataclass
class OrderPlan:
    order_id: str
    kind: str
    seed: int
    schedule: list[list[str]]
    hard_pairs: list[tuple[str, str]] = field(default_factory=list)
    hard_pair_source: str = ""


@dataclass
class StreamPlan:
    admissible: list[str]
    streamed: list[str]
    held_back: list[str]
    permanent_unknown: list[str]
    sizes: list[int]
    departures: list[str]
    orders: list[OrderPlan]
    hard_pair_source: str
    fingerprint: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "StreamPlan":
        orders = [OrderPlan(**{**o, "hard_pairs": [tuple(p) for p in o["hard_pairs"]]}) for o in d["orders"]]
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**{**keep, "orders": orders})

    def order(self, order_id: str) -> OrderPlan:
        for o in self.orders:
            if o.order_id == order_id:
                return o
        raise KeyError(order_id)


def plan_stream(elig: Eligibility, reg: KBRegistry, cfg_schedule: dict, cfg_orders: dict,
                hard_pairs_csv: str | None = None) -> StreamPlan:
    seed = int(cfg_orders["seed"])
    adm = sorted(elig.admissible)
    sizes, n_held, notes = derive_schedule(len(adm), cfg_schedule["initial"], cfg_schedule["batch"],
                                           cfg_schedule["stages"], cfg_schedule.get("min_initial", 4),
                                           cfg_schedule.get("min_batch", 2))
    strata = {c: (reg._get_record(c).disease_type if reg._get_record(c) else "") for c in adm}
    rng = random.Random(seed)
    held = sorted(rng.sample(adm, n_held)) if n_held else []
    streamed = [c for c in adm if c not in held]

    orders: list[OrderPlan] = []
    for k in range(int(cfg_orders["random_orders"])):
        s = seed + 1000 * (k + 1)
        seq = _stratified(streamed, strata, random.Random(s))
        orders.append(OrderPlan(f"O1_r{k}", "stratified_random", s, _cut(seq, sizes)))
    pairs, src = hard_pairs(streamed, reg, cfg_orders.get("hard_pair_source", "registry_look_alikes"),
                            hard_pairs_csv)
    if len(sizes) >= 2 and pairs:
        if cfg_orders.get("confusable", True):
            sch, ch = _confusable_order(sizes, pairs, streamed, strata, random.Random(seed + 7), False)
            orders.append(OrderPlan("O2_confusable", "confusable_after_active", seed + 7, sch,
                                    [tuple(p) for p in ch], src))
        if cfg_orders.get("reversed", True):
            sch, ch = _confusable_order(sizes, pairs, streamed, strata, random.Random(seed + 7), True)
            orders.append(OrderPlan("O3_reversed", "confusable_reversed", seed + 7, sch,
                                    [tuple(p) for p in ch], src))
    plan = StreamPlan(adm, streamed, held, sorted(elig.permanent_unknown), sizes, notes, orders, src)
    plan.fingerprint = stable_hash(plan.to_json())
    return plan
