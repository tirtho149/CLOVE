import pytest

from clove.config import Config
from clove.contracts import (AccessScope, Claim, ClaimKind, Decision, FutureClassLeak,
                             InferenceResult)
from clove.inference.backend import ImageGuard, MockBackend, ReferenceImageViolation
from clove.inference.decode import decode
from clove.inference.prompt import build_prompt
from clove.inference.token_meter import BudgetMeter, RegexTokenizer
from clove.kb.inventory import AuthorizedKB, ClassLedger, inventory_hash
from clove.memory.base import pack_evidence, restore, snapshot
from clove.memory.policy import Query
from clove.memory.snapshot import make_policy
from clove.orchestration.policy import UnknownGate, scored_decision
from clove.resources import load_registry

METER = BudgetMeter(RegexTokenizer())


def scope(ids, t=0):
    return AccessScope(t, frozenset(ids), inventory_hash(ids))


def claim(cid, cls, text, mentions=()):
    return Claim(cid, cls, ClaimKind.QUOTE, text, text, "https://x", "t", "f", tuple(mentions))


def test_pack_evidence_cap_and_whole_claims():
    sc = scope(["A", "B"])
    cl = [claim(f"c{i}", "A", "word " * (5 + i)) for i in range(10)]
    kept, used, dropped = pack_evidence(cl, sc, METER, 40, 96)
    assert used <= 40 and dropped > 0
    assert all(k.text.startswith(f"[{k.claim_id}]") for k in kept)


def test_pack_evidence_raises_on_future_class():
    with pytest.raises(FutureClassLeak):
        pack_evidence([claim("x", "Z", "t")], scope(["A"]), METER, 100, 96)
    with pytest.raises(FutureClassLeak):
        pack_evidence([claim("x", "A", "t", mentions=("Z",))], scope(["A"]), METER, 100, 96)


def _setup(sage_fixture):
    cfg = Config.load(sage_fixture["config"])
    _, reg = load_registry(cfg)
    return cfg, reg, cfg.budget()


def test_append_never_recompiles_fresh_does(sage_fixture):
    cfg, reg, b = _setup(sage_fixture)
    sched = [["Dummy_Alpha_Spot", "Dummy_Gamma_Rot"], ["Dummy_Beta_Blight"]]
    led = ClassLedger(sched)
    app, fr = make_policy("append", cfg, METER), make_policy("fresh", cfg, METER)
    for t in range(2):
        sc = led.advance()
        akb = AuthorizedKB(reg, sc)
        app.update(akb, tuple(sched[t]), b)
        fr.update(akb, tuple(sched[t]), b)
    assert app.state.cards["Dummy_Alpha_Spot"].compiled_episode == 0
    assert fr.state.cards["Dummy_Alpha_Spot"].compiled_episode == 1


def test_masked_mention_enters_memory_only_after_admission(sage_fixture):
    """Alpha's 'Unlike Dummy Beta Blight' sentence is hidden at t=0 and usable at t=1."""
    cfg, reg, b = _setup(sage_fixture)
    sched = [["Dummy_Alpha_Spot", "Dummy_Gamma_Rot"], ["Dummy_Beta_Blight"]]
    led = ClassLedger(sched)
    sc0 = led.advance()
    akb0 = AuthorizedKB(reg, sc0)
    assert all("Dummy_Beta_Blight" not in c.mentions for c in akb0.claims("Dummy_Alpha_Spot"))
    assert akb0.masked_claims["Dummy_Alpha_Spot"] >= 1
    sc1 = led.advance()
    akb1 = AuthorizedKB(reg, sc1)
    assert any("Dummy_Beta_Blight" in c.mentions for c in akb1.claims("Dummy_Alpha_Spot"))


def test_ledger_rejects_unscheduled_admission():
    led = ClassLedger([["A"], ["B"]])
    led.advance()
    with pytest.raises(FutureClassLeak):
        led.admit("B", 0)


def test_snapshot_restore(sage_fixture, tmp_path):
    cfg, reg, b = _setup(sage_fixture)
    led = ClassLedger([["Dummy_Alpha_Spot", "Dummy_Gamma_Rot"]])
    sc = led.advance()
    p = make_policy("prism", cfg, METER)
    p.update(AuthorizedKB(reg, sc), ("Dummy_Alpha_Spot", "Dummy_Gamma_Rot"), b)
    h = snapshot(p.state, tmp_path / "s.json")
    st = restore(tmp_path / "s.json", sc)
    assert st.hash() == h
    with pytest.raises(FutureClassLeak):
        restore(tmp_path / "s.json", scope(["Dummy_Alpha_Spot"]))


@pytest.mark.parametrize("m", ["append", "fresh", "prism", "uniform", "recency", "random", "bm25", "shuffled"])
def test_every_method_respects_kb_cap(sage_fixture, m):
    cfg, reg, b = _setup(sage_fixture)
    from clove.config import Budget
    small = Budget("tiny", 20, 3, 96, 180, 96)
    led = ClassLedger([["Dummy_Alpha_Spot", "Dummy_Gamma_Rot", "Dummy_Beta_Blight"]])
    sc = led.advance()
    p = make_policy(m, cfg, METER)
    p.update(AuthorizedKB(reg, sc), tuple(sorted(sc.allowed_class_ids)), small)
    sel = p.select(Query("img", 0, sc, {}), small)
    assert sel.kb_tokens <= 20
    assert sum(l.tokens for ls in sel.evidence.values() for l in ls) == sel.kb_tokens


def test_decode_logprob_distribution_and_gate():
    from clove.memory.policy import Selection
    sel = Selection(["A_cls", "B_cls"], {"A_cls": [], "B_cls": []}, 0, 0, "h")
    pr = build_prompt(sel)
    toks = ['{"', "answer", '":', ' "', "B", '"}']
    tl = [{t: -0.01, "__token__": t} for t in toks]
    tl[4] = {"A": -2.0, "B": -0.2, "0": -3.0, "__token__": "B"}
    res = InferenceResult('{"answer": "B"}', 100, 6, 0.1, token_logprobs=tl)
    d = decode(res, pr)
    assert d.ok and d.label_id == "B_cls" and d.score_source == "logprobs"
    assert abs(sum(d.probs.values()) + d.p_unknown - 1) < 1e-9
    g = UnknownGate("threshold", 0.95)
    g.calibrate([0.05, 0.1, 0.2, 0.3])
    assert g.threshold == 0.3
    dec, lab, s = g.decide(d)
    assert dec == Decision.KNOWN and lab == "B_cls"
    bad = decode(InferenceResult("I think it's B", 100, 6, 0.1), pr)
    dec, lab, _ = g.decide(bad)
    assert dec == Decision.UNRESOLVED and scored_decision(dec) == "UNKNOWN"


def test_image_guard_enforces_single_declared_query(tmp_path):
    from tests.conftest import make_image
    q, ref = tmp_path / "q.jpg", tmp_path / "ref.jpg"
    make_image(q, 1)
    make_image(ref, 2)
    g = ImageGuard(MockBackend(), forbidden_paths={str(ref)})
    with pytest.raises(ReferenceImageViolation):
        g.infer(str(q), "s", "A) x\n", 8)            # not declared
    with pytest.raises(ReferenceImageViolation):
        g.allow(str(ref))                            # reference image
    g.allow(str(q))
    with pytest.raises(ReferenceImageViolation):
        g.infer(str(q), "s", "see ref.jpg", 8)       # image path in prompt
    r = g.infer(str(q), "s", "A) x\n", 8)
    assert r.synthetic and g.calls == 1 and g.images_loaded == [str(q.resolve())]
