"""clove.inference.claude_headless — Claude Code headless (`claude -p`) backend.

Mirrors the SAGE open_agentic convention (one `claude -p` subprocess per image,
stream-json output, cleaned environment) under the CLOVE contract.

Two image modes:

* ``inline`` (default, recommended): the query photo is sent as a base64 image block
  in a single stream-json user message, and ALL tools are disabled (``--tools ""``).
  The model has no filesystem access at all, so it cannot open a reference image:
  k_img = 0 holds by construction.
* ``read_tool`` (SAGE-compatible): the photo is copied under a neutral name into an
  empty temporary working directory and the model may only use ``Read``. The
  stream-json trace is audited: any Read of a path other than that copy is reported
  as ``REFERENCE_IMAGE_VIOLATION`` and the prediction becomes an error.

Accounting: input tokens = input + cache_creation + cache_read tokens reported in the
final ``result`` event (exact, Anthropic-side, includes image tokens); output tokens
and ``total_cost_usd`` come from the same event. Claude does not expose logprobs, so
option scores are *hard* (one-hot); `score_source = hard` is recorded and unknown
detection reduces to the model's explicit "0 = none of these" choice. Decoding is not
seeded, so repeated calls can differ; the response cache keeps the first answer.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from ..contracts import InferenceResult

ENV_STRIP_PREFIXES = ("CLAUDE", "CURSOR", "MCP_CONNECTION", "VSCODE", "ELECTRON")


def _media_type(b: bytes) -> str:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/jpeg"


def parse_stream_json(stdout: str) -> tuple[list[dict], dict]:
    """NDJSON -> (observable tool/text trace, final result event)."""
    trace: list[dict] = []
    result: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "assistant":
            for c in (obj.get("message") or {}).get("content") or []:
                if c.get("type") == "tool_use":
                    trace.append({"type": "tool_use", "tool": c.get("name"), "input": c.get("input", {})})
                elif c.get("type") == "text" and c.get("text", "").strip():
                    trace.append({"type": "text", "chars": len(c["text"])})
        elif obj.get("type") == "result":
            result = obj
    return trace, result


class ClaudeHeadlessBackend:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        c = cfg.get("claude") or {}
        self.bin = c.get("bin", "claude")
        self.model = c.get("model", "claude-sonnet-5-5")
        self.mode = c.get("image_mode", "inline")
        if self.mode not in ("inline", "read_tool"):
            raise ValueError("claude.image_mode must be inline | read_tool")
        self.effort = c.get("effort")
        self.strip_api_key = bool(c.get("strip_api_key", True))
        self.enforce_output_cap = bool(c.get("enforce_output_cap", True))
        # Claude Code marks prompts for caching; a cache WRITE costs 2x plain input. When the
        # same image does not recur within the cache TTL (the pilot's method-major loop),
        # caching only adds cost. Pricing-only switch: deliberately NOT part of describe()
        # (answers are unaffected), so existing responses stay valid.
        self.prompt_caching = bool(c.get("prompt_caching", True))
        self.extra_args = list(c.get("extra_args", []))
        self.timeout = float(cfg.get("timeout_s", 180))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.name = f"claude_headless:{self.model}:{self.mode}"
        self._version = None

    # ── info ──────────────────────────────────────────────────────────────
    def version(self) -> str:
        if self._version is None:
            try:
                out = subprocess.run([self.bin, "--version"], capture_output=True, text=True,
                                     timeout=30, env=self._env(None))
                self._version = out.stdout.strip() or out.stderr.strip()
            except Exception as e:  # noqa: BLE001
                self._version = f"unavailable: {e}"
        return self._version

    def describe(self) -> dict:
        return {"backend": "claude_headless", "model_id": self.model, "image_mode": self.mode,
                "effort": self.effort, "cli_version": self.version(),
                "enforce_output_cap": self.enforce_output_cap, "logprobs": False,
                "seeded": False, "extra_args": self.extra_args}

    def health(self) -> tuple[bool, str]:
        if shutil.which(self.bin) is None and not Path(self.bin).exists():
            return False, f"`{self.bin}` not found on PATH"
        v = self.version()
        if v.startswith("unavailable"):
            return False, v
        return True, v

    # ── call ──────────────────────────────────────────────────────────────
    def _env(self, max_new_tokens: int | None) -> dict:
        env = {k: v for k, v in os.environ.items() if not k.startswith(ENV_STRIP_PREFIXES)}
        if self.strip_api_key:
            env.pop("ANTHROPIC_API_KEY", None)     # use the logged-in account, as SAGE did
        if not self.prompt_caching:
            env["DISABLE_PROMPT_CACHING"] = "1"
        if max_new_tokens and self.enforce_output_cap:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(int(max_new_tokens))
        return env

    def _cmd(self, system: str, tools: str) -> list[str]:
        cmd = [self.bin, "-p",
               "--input-format", "stream-json",
               "--output-format", "stream-json", "--verbose",
               "--no-session-persistence",
               "--model", self.model,
               "--system-prompt", system,        # REPLACES Claude Code's long default prompt
               "--tools", tools,
               "--setting-sources", "",          # ignore user/project settings, hooks, CLAUDE.md
               "--strict-mcp-config",            # no MCP servers
               "--disable-slash-commands"]
        if tools:
            cmd += ["--allowedTools", tools]
        if self.effort:
            cmd += ["--effort", str(self.effort)]
        return cmd + self.extra_args

    def infer(self, image_path: str, system: str, user_text: str, max_new_tokens: int) -> InferenceResult:
        work = tempfile.mkdtemp(prefix="clove_claude_")
        try:
            if self.mode == "inline":
                raw = Path(image_path).read_bytes()
                content = [{"type": "image", "source": {"type": "base64", "media_type": _media_type(raw),
                                                        "data": base64.b64encode(raw).decode()}},
                           {"type": "text", "text": user_text}]
                tools = ""
                neutral = None
            else:
                neutral = str(Path(work) / f"query{Path(image_path).suffix.lower() or '.jpg'}")
                shutil.copy2(image_path, neutral)
                content = [{"type": "text", "text": f"Read this file to see the photo: `{neutral}`\n\n{user_text}"}]
                tools = "Read"
            msg = json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n"
            err = None
            for attempt in range(self.max_retries + 1):
                res, err, retry = self._once(self._cmd(system, tools), msg, work, max_new_tokens, neutral)
                if res is not None:
                    return res
                if not retry:
                    break
                time.sleep(min(30, 2 ** attempt * 2))
            return InferenceResult("", 0, 0, 0.0, error=err)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _once(self, cmd, msg, cwd, max_new_tokens, neutral):
        t0 = time.time()
        proc = None
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=self._env(max_new_tokens), cwd=cwd,
                                    start_new_session=True)
            out, errb = proc.communicate(input=msg.encode(), timeout=self.timeout)
        except subprocess.TimeoutExpired:
            if proc:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            return None, f"TIMEOUT after {self.timeout}s", True
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}", False
        dt = time.time() - t0
        trace, result = parse_stream_json(out.decode(errors="replace"))
        if proc.returncode != 0 or not result or result.get("is_error"):
            detail = (result.get("result") if result else None) or errb.decode(errors="replace")[:400]
            transient = any(s in str(detail).lower() for s in ("overloaded", "rate limit", "529", "timeout", "503"))
            err = f"claude exit {proc.returncode}: {detail}"
            if "output token maximum" in str(detail):
                err = f"OUTPUT_CAP_EXCEEDED: {detail}"
            if result and not transient:
                # failed calls are still billed: keep their usage and cost
                n_in, n_out, meta = self._usage(result, trace)
                return InferenceResult("", n_in, n_out, dt, error=err, cost_usd=result.get("total_cost_usd"),
                                       meta=meta), None, False
            return None, err, transient
        reads = [t for t in trace if t["type"] == "tool_use"]
        if neutral is None and reads:
            return None, f"REFERENCE_IMAGE_VIOLATION: tool use with tools disabled: {reads}", False
        if neutral is not None:
            bad = [t for t in reads if t["tool"] != "Read"
                   or os.path.realpath(str(t["input"].get("file_path", ""))) != os.path.realpath(neutral)]
            if bad:
                return None, f"REFERENCE_IMAGE_VIOLATION: {bad[:3]}", False
        n_in, n_out, meta = self._usage(result, trace)
        return InferenceResult(str(result.get("result") or ""), n_in, n_out, dt,
                               image_tokens=None, token_logprobs=None,
                               cost_usd=result.get("total_cost_usd"), meta=meta), None, False

    @staticmethod
    def _usage(result: dict, trace: list[dict]) -> tuple[int, int, dict]:
        u = result.get("usage") or {}
        n_in = int(u.get("input_tokens", 0) or 0) + int(u.get("cache_creation_input_tokens", 0) or 0) \
            + int(u.get("cache_read_input_tokens", 0) or 0)
        meta = {"num_turns": result.get("num_turns"), "duration_ms": result.get("duration_ms"),
                "duration_api_ms": result.get("duration_api_ms"),
                "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": u.get("cache_read_input_tokens"),
                "uncached_input_tokens": u.get("input_tokens"),
                "tool_uses": sum(1 for t in trace if t["type"] == "tool_use"),
                "model_usage": result.get("modelUsage")}
        return n_in, int(u.get("output_tokens", 0) or 0), meta
