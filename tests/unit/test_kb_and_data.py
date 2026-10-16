import pytest

from clove.config import Config
from clove.contracts import ClaimKind
from clove.data.splits import derive_schedule
from clove.kb.reader import ClassNamer, build_registry, read_xlsx_rows
from clove.resources import build_canonical_manifest, build_plan, load_registry


def test_xlsx_reader_stdlib(sage_fixture):
    rows = read_xlsx_rows(sage_fixture["cfg"]["paths"]["kb_xlsx"])
    assert rows[0][0] == "Disease" and rows[1][0] == "Dummy_Alpha_Spot"


def test_namer_aliases_and_case():
    n = ClassNamer(["Rust", "Brown_Spot"], {"Soybean_rust": "Rust", "Septoria brown spot": "Brown_Spot"})
    assert n.resolve("soybean_Rust") == "Rust"
    assert n.resolve("Septoria_Brown_Spot") == "Brown_Spot"
    assert n.resolve("brown-spot") == "Brown_Spot"
    assert n.resolve("Nope") is None


def test_registry_claim_kinds_and_mentions(sage_fixture):
    cfg = Config.load(sage_fixture["config"])
    _, reg = load_registry(cfg)
    kinds = reg.kind_counts("Dummy_Alpha_Spot")
    assert kinds.get("quote", 0) >= 1 and kinds.get("kb_description", 0) >= 1
    assert kinds.get("look_alike") == 1
    alpha = reg._get_claims("Dummy_Alpha_Spot")
    assert any("Dummy_Beta_Blight" in c.mentions for c in alpha)
    rec = reg._get_record("Dummy_Alpha_Spot")
    assert rec.look_alikes == ("Dummy_Beta_Blight",)


def test_pathogen_binomial_not_a_mention(tmp_path):
    from tests.conftest import write_xlsx
    p = tmp_path / "k.xlsx"
    write_xlsx(p, [["Disease", "Pathogen", "Type of Disease", "Affected Parts", "Visual Description"],
                   ["Frogeye", "Cercospora sojina", "Fungal", "Foliar", "Caused by Cercospora sojina; gray spots."],
                   ["Cercospora", "Cercospora kikuchii", "Fungal", "Foliar", "Purple leaves."]])
    reg = build_registry(ClassNamer(["Frogeye", "Cercospora"]), kb_xlsx=str(p))
    assert all(not c.mentions for c in reg._get_claims("Frogeye"))


@pytest.mark.parametrize("n,expect,held", [
    (26, [10, 5, 5, 5], 1), (25, [10, 5, 5, 5], 0), (23, [8, 5, 5, 5], 0), (16, [4, 4, 4, 4], 0)])
def test_derive_schedule(n, expect, held):
    sizes, h, notes = derive_schedule(n, 10, 5, 4)
    assert sizes == expect and h == held
    if n != 25:
        assert notes


def test_derive_schedule_too_few():
    with pytest.raises(ValueError):
        derive_schedule(3, 10, 5, 4)


def test_manifest_and_plan_deterministic(sage_fixture):
    cfg = Config.load(sage_fixture["config"])
    namer, reg = load_registry(cfg)
    m1 = build_canonical_manifest(cfg, namer)
    m2 = build_canonical_manifest(cfg, namer)
    assert m1.fingerprint == m2.fingerprint
    p1, _ = build_plan(cfg, m1, reg)
    p2, _ = build_plan(cfg, m2, reg)
    assert p1.fingerprint == p2.fingerprint
    assert "Dummy_Zeta_Injury" in p1.permanent_unknown
    for o in p1.orders:
        flat = [c for b in o.schedule for c in b]
        assert sorted(flat) == sorted(p1.streamed) and len(flat) == len(set(flat))
    assert {o.kind for o in p1.orders} >= {"stratified_random", "confusable_after_active"}


def test_dev_images_never_overlap_test(sage_fixture):
    cfg = Config.load(sage_fixture["config"])
    namer, _ = load_registry(cfg)
    man = build_canonical_manifest(cfg, namer)
    test_g = {r.group_id for r in man.images if r.split == "test"}
    assert not any(r.group_id in test_g for r in man.images if r.split == "dev")


def test_claim_kind_enum_roundtrip():
    assert ClaimKind("quote") is ClaimKind.QUOTE
