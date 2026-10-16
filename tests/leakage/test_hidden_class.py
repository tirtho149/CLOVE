"""Leakage invariant (protocol §11) with a deliberately hidden synthetic class."""
import pytest

from clove.cli import leakage_checks, main
from clove.config import Config
from clove.contracts import AccessScope, FutureClassLeak, assert_in_scope
from clove.kb.inventory import AuthorizedKB, inventory_hash
from clove.resources import load_registry


def test_leakage_checks_pass_on_fixture(sage_fixture):
    c = sage_fixture["config"]
    assert main(["manifest", "--config", c]) == 0
    res = leakage_checks(Config.load(c))
    assert not res["violations"], res["violations"]
    assert res["checks"]["hidden_class_kb_request_rejected"]["ok"]


def test_hidden_class_never_reachable(sage_fixture):
    cfg = Config.load(sage_fixture["config"])
    _, reg = load_registry(cfg)
    hidden = "Dummy_Beta_Blight"
    ids = ["Dummy_Alpha_Spot", "Dummy_Gamma_Rot"]
    sc = AccessScope(0, frozenset(ids), inventory_hash(ids))
    akb = AuthorizedKB(reg, sc)
    with pytest.raises(FutureClassLeak):
        akb.claims(hidden)
    with pytest.raises(FutureClassLeak):
        akb.record(hidden)
    for cid in ids:
        for cl in akb.claims(cid):
            assert hidden not in cl.mentions
    with pytest.raises(FutureClassLeak):
        assert_in_scope(sc, [], hidden)
    strict = AuthorizedKB(reg, sc, mask_mentions=False)
    with pytest.raises(FutureClassLeak):
        strict.claims("Dummy_Alpha_Spot")   # its claim names the hidden class
