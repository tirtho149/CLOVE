"""End-to-end CPU pipeline on the dummy fixture with the MOCK backend.

Validates wiring, resumability, determinism and leakage — never results."""
import json
import re
from pathlib import Path

from clove.cli import main
from clove.config import Config
from clove.resources import load_plan


def _run(cfg_path, *args):
    return main([args[0], "--config", cfg_path, *args[1:]])


def _prep(fx):
    c = fx["config"]
    assert _run(c, "audit") == 0
    assert _run(c, "manifest") == 0
    assert _run(c, "test-leakage") == 0
    return c


def test_full_synthetic_pipeline_resume_and_determinism(sage_fixture):
    c = _prep(sage_fixture)
    art = Path(sage_fixture["cfg"]["paths"]["artifacts"])
    assert _run(c, "smoke", "--synthetic") == 0
    assert _run(c, "run", "--synthetic", "--run-id", "r1") == 0
    p1 = art / "synthetic_runs" / "r1" / "predictions.jsonl"
    n1 = len(p1.read_text().splitlines())
    assert n1 > 0
    # resume: nothing new is computed
    assert _run(c, "run", "--synthetic", "--run-id", "r1") == 0
    st = json.loads((art / "synthetic_runs" / "r1" / "status.json").read_text())
    assert st["detail"]["counters"]["new_calls"] == 0
    assert len(p1.read_text().splitlines()) == n1
    # determinism: an independent run produces identical predictions
    assert _run(c, "run", "--synthetic", "--run-id", "r2") == 0
    strip = lambda p: sorted(json.dumps({k: v for k, v in json.loads(l).items()
                                         if k not in ("run_id", "latency_s", "raw_response_path", "cached")},
                                        sort_keys=True) for l in p.read_text().splitlines())
    assert strip(p1) == strip(art / "synthetic_runs" / "r2" / "predictions.jsonl")
    assert main(["analyze", "--run-dir", str(art / "synthetic_runs" / "r1")]) == 0
    rep = (art / "synthetic_runs" / "r1" / "reports" / "findings.md").read_text()
    assert "SYNTHETIC — NOT RESULTS" in rep


def test_prompts_never_name_unadmitted_classes(sage_fixture):
    c = _prep(sage_fixture)
    art = Path(sage_fixture["cfg"]["paths"]["artifacts"])
    assert _run(c, "run", "--synthetic", "--run-id", "leak") == 0
    plan = load_plan(Config.load(c))
    rd = art / "synthetic_runs" / "leak"
    recs = [json.loads(l) for l in (rd / "predictions.jsonl").read_text().splitlines()]
    sched = {o.order_id: o.schedule for o in plan.orders}
    pool = set(plan.streamed) | set(plan.permanent_unknown) | set(plan.held_back)
    checked = 0
    for r in recs:
        admitted = {x for b in sched[r["order_id"]][: r["episode"] + 1] for x in b}
        prompt = json.loads((rd / "prompts" / f"{r['prompt_hash']}.json").read_text())
        text = prompt["system"] + prompt["user"]
        for cls in pool - admitted:
            for form in {cls, cls.replace("_", " ")}:
                assert not re.search(re.escape(form), text, re.I), (cls, r["episode"], r["method"])
        assert set(r["candidates"]) <= admitted
        assert r["label_id"] is None or r["label_id"] in admitted
        checked += 1
    assert checked > 0


def test_refuses_real_run_without_leakage_and_estimate(sage_fixture):
    c = sage_fixture["config"]
    assert _run(c, "audit") == 0
    assert _run(c, "manifest") == 0
    assert _run(c, "run") == 2          # no leakage report yet
    assert _run(c, "test-leakage") == 0
    assert _run(c, "run") == 2          # no compute estimate


def test_manifest_freeze_is_immutable(sage_fixture):
    c = _prep(sage_fixture)
    import yaml
    cfg = yaml.safe_load(Path(c).read_text())
    cfg["orders"]["seed"] = 99
    Path(c).write_text(yaml.safe_dump(cfg))
    assert _run(c, "manifest") == 2


def test_mock_backend_refused_outside_synthetic(sage_fixture):
    c = _prep(sage_fixture)
    import yaml
    cfg = yaml.safe_load(Path(c).read_text())
    cfg["model"]["backend"] = "mock"
    Path(c).write_text(yaml.safe_dump(cfg))
    assert _run(c, "run", "--skip-estimate") == 2


def test_unreachable_backend_is_blocked_not_faked(sage_fixture):
    c = _prep(sage_fixture)
    import yaml
    cfg = yaml.safe_load(Path(c).read_text())
    cfg["model"]["base_url"] = "http://127.0.0.1:9"
    Path(c).write_text(yaml.safe_dump(cfg))
    assert _run(c, "smoke") == 2
    art = Path(cfg["paths"]["artifacts"])
    st = json.loads(next((art / "runs").glob("smoke_*/status.json")).read_text())
    assert st["status"] == "BLOCKED" and st["resume_command"]


def test_claude_headless_pipeline_with_fake_cli(sage_fixture, tmp_path):
    """smoke -> estimate (USD) -> run -> analyze on the claude_headless backend with a
    stand-in CLI: checks wiring, cost accounting, parallel workers and the budget gate."""
    import sys
    import yaml
    c = _prep(sage_fixture)
    fake = Path(__file__).resolve().parents[1] / "fixtures" / "fake_claude.py"
    wrapper = tmp_path / "claude"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {fake} \"$@\"\n")
    wrapper.chmod(0o755)
    cfg = yaml.safe_load(Path(c).read_text())
    cfg["model"] = {"backend": "claude_headless", "model_id": "claude-test", "tokenizer": "approx",
                    "require_exact_tokenizer": False, "timeout_s": 30, "max_retries": 0,
                    "claude": {"bin": str(wrapper), "model": "claude-test"}}
    cfg["gate"] = {"mode": "argmax_only", "dev_per_class": 2}
    cfg["compute"] = {"parallel": 4, "max_usd": 1000.0}
    Path(c).write_text(yaml.safe_dump(cfg))
    assert _run(c, "smoke") == 0
    assert _run(c, "estimate") == 0
    est = json.loads((Path(cfg["paths"]["artifacts"]) / "reports" / "compute_estimate.json").read_text())
    assert est["backend"] == "claude_headless" and 0 < est["usd_per_call"] <= 0.0011 and est["cost_model"]["n_calls"] >= 8 and est["within_budget"]
    assert _run(c, "run", "--run-id", "cl") == 0
    rd = Path(cfg["paths"]["artifacts"]) / "runs" / "cl"
    recs = [json.loads(l) for l in (rd / "predictions.jsonl").read_text().splitlines()]
    assert recs and all(not r["synthetic"] for r in recs)
    assert all(r["extra"]["score_source"] == "hard" for r in recs if r["failure"] != "OUTPUT_PARSE")
    assert all(r["extra"]["tokenizer_exact"] is False for r in recs)
    assert main(["analyze", "--run-dir", str(rd)]) == 0
    # USD ceiling stops cleanly with PARTIAL + resume command. Cache hits cost nothing,
    # so use a method whose prompts were never sent (forces new calls).
    cfg["compute"]["max_usd"] = 0.0005
    Path(c).write_text(yaml.safe_dump(cfg))
    assert _run(c, "run", "--skip-estimate", "--run-id", "cl_capped", "--methods", "uniform") == 1
    st = json.loads((Path(cfg["paths"]["artifacts"]) / "runs" / "cl_capped" / "status.json").read_text())
    assert st["status"] == "PARTIAL" and st["resume_command"]
