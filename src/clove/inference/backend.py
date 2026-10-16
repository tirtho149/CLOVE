"""clove.inference.backend — frozen VLM backends behind one interface.

* ``openai_compat`` — a vLLM (or any OpenAI-compatible) server hosting the frozen
  checkpoint. Exact prompt/completion token counts come from the server's `usage`
  (prompt tokens INCLUDE image tokens); top-logprobs give option scores.
* ``transformers`` — in-process Hugging Face model (Qwen3-VL), eval mode, gradients
  disabled, weight fingerprint checked before/after a run.
* ``claude_headless`` — Claude Code ``claude -p`` subprocess per image
  (see ``inference/claude_headless.py``): exact usage + cost, no logprobs.
* ``mock`` — deterministic synthetic backend for CPU tests ONLY; every result it
  returns is flagged ``synthetic=True`` and the CLI refuses to report it as data.

The image is sent as bytes (never as a path), so class names in file paths cannot leak.
`ImageGuard` enforces k_img = 0: one image per call, and only the current query image.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Protocol

from ..contracts import InferenceResult


class ReferenceImageViolation(PermissionError):
    pass


class VLMBackend(Protocol):
    name: str

    def infer(self, image_path: str, system: str, user_text: str, max_new_tokens: int
              ) -> InferenceResult: ...

    def describe(self) -> dict: ...


class ImageGuard:
    """Wraps a backend; the orchestrator declares the single permitted query image
    before each call. Any other image (e.g. a SAGE reference image) raises."""

    def __init__(self, backend: VLMBackend, forbidden_paths: set[str] | None = None):
        self.backend = backend
        self.name = backend.name
        self.forbidden = {os.path.realpath(p) for p in (forbidden_paths or set())}
        self._local = threading.local()          # one declared query image per worker thread
        self._lock = threading.Lock()
        self.calls = 0
        self.images_loaded: list[str] = []

    def allow(self, image_path: str) -> None:
        rp = os.path.realpath(image_path)
        if rp in self.forbidden:
            raise ReferenceImageViolation(f"{image_path} is a dev/reference image; k_img=0")
        self._local.allowed = rp

    def infer(self, image_path: str, system: str, user_text: str, max_new_tokens: int) -> InferenceResult:
        rp = os.path.realpath(image_path)
        allowed = getattr(self._local, "allowed", None)
        if allowed is None or rp != allowed:
            raise ReferenceImageViolation("image not declared as the current query image")
        if re.search(r"\.(jpe?g|png|webp|bmp)\b", system + user_text, re.I):
            raise ReferenceImageViolation("prompt text references an image file")
        self._local.allowed = None               # one call per declaration
        with self._lock:
            self.calls += 1
            self.images_loaded.append(rp)
        return self.backend.infer(image_path, system, user_text, max_new_tokens)

    def describe(self) -> dict:
        return self.backend.describe()


def _data_url(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    mime = "image/png" if b[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(b).decode()}"


class OpenAICompatBackend:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = f"openai_compat:{cfg['model_id']}"
        # One or more identical servers (e.g. one vLLM per GPU); requests round-robin.
        bases = cfg.get("base_urls") or [cfg["base_url"]]
        self.bases = [b.rstrip("/") for b in bases]
        self._rr = 0
        self._rr_lock = threading.Lock()
        self.key = os.environ.get(cfg["api_key_env"]) if cfg.get("api_key_env") else None

    def describe(self) -> dict:
        # server addresses are deliberately excluded: identical replicas must share the cache
        return {"backend": "openai_compat", "model_id": self.cfg["model_id"],
                "revision": self.cfg.get("revision"), "temperature": self.cfg.get("temperature", 0.0),
                "top_logprobs": self.cfg.get("top_logprobs", 10), "dtype": self.cfg.get("dtype"),
                "max_pixels": self.cfg.get("max_pixels")}

    def _next_url(self) -> str:
        with self._rr_lock:
            b = self.bases[self._rr % len(self.bases)]
            self._rr += 1
        return b + "/v1/chat/completions"

    def health(self) -> tuple[bool, str]:
        for b in self.bases:
            try:
                req = urllib.request.Request(b + "/v1/models", headers=self._headers())
                with urllib.request.urlopen(req, timeout=10) as r:
                    models = [m["id"] for m in json.loads(r.read())["data"]]
                if self.cfg["model_id"] not in models:
                    return False, f"{b}: server up but model {self.cfg['model_id']!r} not in {models}"
            except Exception as e:  # noqa: BLE001
                return False, f"{b}: {type(e).__name__}: {e}"
        return True, f"ok ({len(self.bases)} server(s))"

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def infer(self, image_path, system, user_text, max_new_tokens):
        body = {
            "model": self.cfg["model_id"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
                    {"type": "text", "text": user_text}]},
            ],
            "max_tokens": int(max_new_tokens),
            "temperature": float(self.cfg.get("temperature", 0.0)),
            "logprobs": True,
            "top_logprobs": int(self.cfg.get("top_logprobs", 10)),
            "seed": 0,
        }
        data = json.dumps(body).encode()
        err = None
        for attempt in range(int(self.cfg.get("max_retries", 2)) + 1):
            t0 = time.time()
            try:
                req = urllib.request.Request(self._next_url(), data=data, headers=self._headers())
                with urllib.request.urlopen(req, timeout=float(self.cfg.get("timeout_s", 180))) as r:
                    out = json.loads(r.read())
                dt = time.time() - t0
                ch = out["choices"][0]
                usage = out.get("usage") or {}
                tl = None
                lp = (ch.get("logprobs") or {}).get("content")
                if lp:
                    tl = []
                    for pos in lp:
                        d = {t["token"]: t["logprob"] for t in pos.get("top_logprobs", [])}
                        d.setdefault(pos["token"], pos["logprob"])
                        d["__token__"] = pos["token"]
                        tl.append(d)
                return InferenceResult(ch["message"].get("content") or "",
                                       int(usage.get("prompt_tokens", -1)),
                                       int(usage.get("completion_tokens", -1)), dt,
                                       image_tokens=None, token_logprobs=tl)
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read()[:300]!r}"
                if 400 <= e.code < 500 and e.code != 429:
                    break
            except TimeoutError as e:
                err = f"TIMEOUT: {e}"
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            time.sleep(min(30, 2 ** attempt))
        return InferenceResult("", 0, 0, 0.0, error=err)


class TransformersBackend:
    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.cfg = cfg
        self.name = f"transformers:{cfg['model_id']}"
        pk = {"max_pixels": int(cfg["max_pixels"])} if cfg.get("max_pixels") else {}
        self.proc = AutoProcessor.from_pretrained(cfg["model_id"], revision=cfg.get("revision"), **pk)
        # Turing/Volta (e.g. Titan RTX) have no bf16: set model.dtype: float16 there
        dtype = {"auto": "auto", "float16": torch.float16, "half": torch.float16,
                 "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg.get("dtype") or "auto"]
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg["model_id"], revision=cfg.get("revision"), torch_dtype=dtype, device_map="auto").eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._fp0 = self.weights_fingerprint()

    def describe(self) -> dict:
        return {"backend": "transformers", "model_id": self.cfg["model_id"],
                "revision": self.cfg.get("revision"), "temperature": 0.0,
                "weights_fingerprint": self._fp0}

    def weights_fingerprint(self, n: int = 8) -> str:
        h = hashlib.sha256()
        for i, (name, p) in enumerate(self.model.named_parameters()):
            if i % max(1, sum(1 for _ in self.model.parameters()) // n) == 0:
                h.update(name.encode())
                h.update(str(float(p.detach().float().sum().item())).encode())
        return h.hexdigest()[:16]

    def assert_frozen(self) -> None:
        if self.weights_fingerprint() != self._fp0:
            raise RuntimeError("model weights changed during evaluation")

    def infer(self, image_path, system, user_text, max_new_tokens):
        from PIL import Image
        torch = self.torch
        t0 = time.time()
        try:
            with Image.open(image_path) as im:
                img = im.convert("RGB")
            msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]}]
            text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inp = self.proc(text=[text], images=[img], return_tensors="pt").to(self.model.device)
            n_in = int(inp["input_ids"].shape[1])
            img_tok = None
            tid = getattr(self.proc, "image_token_id", None) or getattr(getattr(self.proc, "tokenizer", None), "image_token_id", None)
            if tid is not None:
                img_tok = int((inp["input_ids"] == tid).sum().item())
            with torch.inference_mode():
                out = self.model.generate(**inp, max_new_tokens=int(max_new_tokens), do_sample=False,
                                          output_scores=True, return_dict_in_generate=True)
            gen = out.sequences[0, n_in:]
            tok = self.proc.tokenizer
            tl = []
            k = int(self.cfg.get("top_logprobs", 10))
            for step, scores in enumerate(out.scores):
                lp = torch.log_softmax(scores[0].float(), dim=-1)
                top = torch.topk(lp, k)
                d = {tok.decode([int(i)]): float(v) for v, i in zip(top.values, top.indices)}
                g = int(gen[step]) if step < len(gen) else None
                if g is not None:
                    gt = tok.decode([g])
                    d.setdefault(gt, float(lp[g]))
                    d["__token__"] = gt
                tl.append(d)
            text_out = tok.decode(gen, skip_special_tokens=True)
            return InferenceResult(text_out, n_in, int(len(gen)), time.time() - t0,
                                   image_tokens=img_tok, token_logprobs=tl)
        except torch.cuda.OutOfMemoryError as e:  # type: ignore[attr-defined]
            return InferenceResult("", 0, 0, time.time() - t0, error=f"OOM: {e}")
        except Exception as e:  # noqa: BLE001
            return InferenceResult("", 0, 0, time.time() - t0, error=f"{type(e).__name__}: {e}")


class MockBackend:
    """Synthetic backend for tests. Given `truth` {image_path: class_id} it answers the
    true class's option with probability `acc` (else a random option), and emits
    logprob-shaped output. NEVER a source of results."""
    name = "mock"

    def __init__(self, truth: dict[str, str] | None = None, acc: float = 0.8, seed: int = 0):
        self.truth = {os.path.realpath(k): v for k, v in (truth or {}).items()}
        self.acc = acc
        self.seed = seed

    def describe(self) -> dict:
        return {"backend": "mock", "synthetic": True, "acc": self.acc, "seed": self.seed}

    def infer(self, image_path, system, user_text, max_new_tokens):
        rng = random.Random(f"{self.seed}:{os.path.realpath(image_path)}:{hashlib.sha1(user_text.encode()).hexdigest()}")
        truth = self.truth.get(os.path.realpath(image_path))
        opts = re.findall(r"^([A-Z]{1,2})\) (.+)$", user_text, re.M)
        if not opts:                                # sage0 style
            classes = re.findall(r"^- (\S+)$", user_text, re.M)
            pred = truth if truth in classes and rng.random() < self.acc else (rng.choice(classes) if classes else "UNKNOWN")
            txt = f'```json\n{{"prediction": "{pred}", "confidence": 0.5, "reasoning": "mock"}}\n```'
            return InferenceResult(txt, len(user_text) // 4 + 300, 30, 0.01, image_tokens=256, synthetic=True)
        codes = [c for c, _ in opts] + ["0"]
        names = {c: n.strip().replace(" ", "_") for c, n in opts}
        true_code = next((c for c, n in names.items() if n == truth), "0")
        ans = true_code if rng.random() < self.acc else rng.choice(codes)
        logits = {c: rng.gauss(0, 1) for c in codes}
        logits[ans] = max(logits.values()) + 1.5 + rng.random()
        z = math.log(sum(math.exp(v) for v in logits.values()))
        toks = ['{"', "answer", '":', ' "', ans, '"}']
        tl = [{t: -0.01, "__token__": t} for t in toks]
        tl[4] = {c: logits[c] - z for c in codes}
        tl[4]["__token__"] = ans
        return InferenceResult(f'{{"answer": "{ans}"}}', len(user_text) // 4 + 300, len(toks), 0.01,
                               image_tokens=256, token_logprobs=tl, synthetic=True)


def make_backend(model_cfg: dict, truth: dict[str, str] | None = None) -> VLMBackend:
    kind = model_cfg.get("backend")
    if kind == "openai_compat":
        return OpenAICompatBackend(model_cfg)
    if kind == "transformers":
        return TransformersBackend(model_cfg)
    if kind == "claude_headless":
        from .claude_headless import ClaudeHeadlessBackend
        return ClaudeHeadlessBackend(model_cfg)
    if kind == "mock":
        return MockBackend(truth, float(model_cfg.get("mock_acc", 0.8)), int(model_cfg.get("mock_seed", 0)))
    raise ValueError(f"unknown backend {kind}")
