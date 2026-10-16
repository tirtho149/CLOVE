import json
import sys
from pathlib import Path

import pytest

from clove.inference.claude_headless import ClaudeHeadlessBackend, parse_stream_json

FAKE = str(Path(__file__).resolve().parents[1] / "fixtures" / "fake_claude.py")


def backend(tmp_path, mode="inline", **kw):
    wrapper = tmp_path / "claude"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n")
    wrapper.chmod(0o755)
    return ClaudeHeadlessBackend({"claude": {"bin": str(wrapper), "model": "claude-test", "image_mode": mode, **kw},
                                  "timeout_s": 30, "max_retries": 0})


@pytest.fixture()
def img(tmp_path):
    from tests.conftest import make_image
    p = tmp_path / "q.jpg"
    make_image(p, 3)
    return str(p)


def test_inline_mode_sends_image_block_and_disables_tools(tmp_path, img, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("CLAUDECODE", "1")
    b = backend(tmp_path)
    assert b.health()[0]
    r = b.infer(img, "sys", "Options:\nA) X\n0) none\nJSON:", 96)
    assert r.error is None and json.loads(r.text) == {"answer": "A"}
    assert r.input_tokens == 1503 and r.output_tokens == 9 and r.cost_usd == 0.001
    call = json.loads(log.read_text().splitlines()[-1])
    assert call["kinds"] == ["image", "text"]
    argv = call["argv"]
    assert argv[argv.index("--tools") + 1] == "" and "--no-session-persistence" in argv
    assert argv[argv.index("--system-prompt") + 1] == "sys"
    assert not call["has_api_key"]
    assert call["env_claude"] == ["CLAUDE_CODE_MAX_OUTPUT_TOKENS"]   # CLAUDECODE etc. stripped


def test_read_tool_mode_allows_only_the_neutral_copy(tmp_path, img, monkeypatch):
    b = backend(tmp_path, mode="read_tool")
    r = b.infer(img, "sys", "Options:\nA) X\n0) none\nJSON:", 96)
    assert r.error is None and r.meta["tool_uses"] == 1
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "read_other")
    r = b.infer(img, "sys", "Options:\nA) X\n0) none\nJSON:", 96)
    assert r.error and r.error.startswith("REFERENCE_IMAGE_VIOLATION")


def test_inline_mode_rejects_any_tool_use(tmp_path, img, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "tool_inline")
    r = backend(tmp_path).infer(img, "sys", "Options:\nA) X\n0) none\nJSON:", 96)
    assert r.error and "REFERENCE_IMAGE_VIOLATION" in r.error


def test_errors_are_reported_not_faked(tmp_path, img, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "error")
    r = backend(tmp_path).infer(img, "sys", "A) X\n", 96)
    assert r.error and r.text == "" and r.input_tokens == 0


def test_parse_stream_json_ignores_garbage():
    tr, res = parse_stream_json('noise\n{"type":"result","result":"ok"}\n')
    assert res["result"] == "ok" and tr == []


def test_output_cap_error_keeps_cost(tmp_path, img, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "cap")
    r = backend(tmp_path).infer(img, "sys", "A) X\n", 96)
    assert r.error.startswith("OUTPUT_CAP_EXCEEDED") and r.cost_usd == 0.05 and r.input_tokens == 16000


def test_prompt_caching_switch_sets_env_without_changing_identity(tmp_path, img, monkeypatch):
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    on, off = backend(tmp_path), backend(tmp_path, prompt_caching=False)
    assert on.describe() == off.describe()
    off.infer(img, "sys", "A) X\n", 96)
    assert "DISABLE_PROMPT_CACHING" in json.loads(log.read_text().splitlines()[-1])["env_all"]
