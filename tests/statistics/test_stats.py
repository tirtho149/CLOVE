import math

import numpy as np
import pytest

from clove.evaluation import continual as C
from clove.evaluation import metrics as M
from clove.evaluation import statistics as S


def test_auroc_known_values():
    assert M.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert M.auroc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert M.auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert M.auroc([], [0.1]) is None


def test_average_precision_and_fpr():
    assert M.average_precision([0.9, 0.8], [0.1]) == 1.0
    assert M.fpr_at_tpr([0.1] * 19 + [0.5], [0.05, 0.9], 0.95) == 0.5
    assert M.fpr_at_tpr([0.1], []) is None


def test_mcnemar_exact():
    r = S.mcnemar_exact([1] * 10 + [0] * 2, [0] * 10 + [1] * 2)
    expected = 2 * sum(math.comb(12, i) for i in range(3)) / 2 ** 12
    assert r["n01"] == 10 and r["n10"] == 2 and abs(r["p"] - expected) < 1e-12


def test_holm_and_bh():
    p = {"a": 0.01, "b": 0.04, "c": 0.03}
    h = S.holm(p)
    assert h == {"a": 0.03, "c": 0.06, "b": 0.06}
    bh = S.benjamini_hochberg(p)
    assert bh["a"] == pytest.approx(0.03) and bh["b"] == pytest.approx(0.04) and bh["c"] == pytest.approx(0.04)


def _units(diff, n_cl=20, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for c in range(n_cl):
        for k in range(3):
            a = float(rng.random() < 0.5 + diff)
            b = float(rng.random() < 0.5)
            out.append(S.PairedUnit("o", f"g{c}", f"cls{c % 4}", a, b, 1 + k % 2))
    return out


def test_bootstrap_deterministic_and_covers_estimate():
    u = _units(0.2)
    b1 = S.paired_cluster_bootstrap(u, n_boot=300, seed=1)
    b2 = S.paired_cluster_bootstrap(u, n_boot=300, seed=1)
    assert b1 == b2
    assert b1["lo"] <= b1["estimate"] <= b1["hi"]


def test_sign_flip_exact_small():
    u = [S.PairedUnit("o", f"g{i}", "c", 1.0, 0.0) for i in range(5)]
    r = S.sign_flip_test(u)
    assert r["method"] == "exact" and r["p"] == pytest.approx(2 / 32)


def test_noninferiority():
    assert S.noninferiority(-0.01, 0.02)["noninferior"] is True
    assert S.noninferiority(-0.03, 0.02)["noninferior"] is False


def rec(t, cls, status, ok, method="m", img=None, tok=100, admit=0):
    return {"episode": t, "gt_class": cls, "gt_status": status, "decision": "KNOWN" if ok else "UNKNOWN",
            "label_id": cls if ok else None, "image_id": img or f"{cls}-x", "variant_id": "orig",
            "prompt_tokens": tok, "output_tokens": 5, "kb_tokens": 50, "vlm_calls": 1, "latency_s": 0.1,
            "gt_admit_episode": admit, "unknown_score": 0.1, "key": f"{t}{cls}{img}", "group_id": "g",
            "method": method}


def test_forgetting_and_bwt():
    rs = [rec(0, "A", "new", True), rec(1, "A", "old", True), rec(2, "A", "old", False)]
    f = C.forgetting(rs)
    assert f["per_class"]["A"] == 1.0 and f["bwt"] == -1.0


def test_memory_forgetting_sign():
    fresh = [rec(1, "A", "old", True, "fresh")]
    pers = [rec(1, "A", "old", False, "append")]
    mf = C.memory_forgetting(fresh, pers)
    assert mf[1]["F_memory"] == 1.0


def test_compute_forgetting_conditional_and_unconditional():
    rs = [rec(0, "A", "new", True, tok=100), rec(1, "A", "old", True, tok=150), rec(2, "A", "old", False, tok=300)]
    cf = C.compute_forgetting(rs)
    assert cf[1]["F_tok_correct_mean"] == pytest.approx(50 / 105)
    assert cf[2]["F_tok_correct_mean"] is None and cf[2]["n_all"] == 1


def test_acquisition_endpoints_separate():
    rs = [rec(0, "B", "pending_unknown", False, admit=1), rec(1, "B", "new", True, admit=1)]
    a = C.acquisition(rs)["B"]
    assert a["Q_pre"] == 1.0 and a["Q_post"] == 1.0


def test_unknown_metrics_na_without_unknowns():
    u = M.unknown_detection([rec(0, "A", "new", True)])
    assert u["auroc"] is None and u["no_eligible_unknowns"]
