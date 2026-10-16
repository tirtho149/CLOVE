"""clove.orchestration.trace — prediction store, response cache and tool trace."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..contracts import InferenceResult
from ..util import JsonlStore, atomic_write_json, atomic_write_text, stable_hash


class ResponseCache:
    """Caches raw backend responses ONLY under byte-identical inputs: image sha256,
    full prompt hash, backend description (model, revision, decoding), max_new_tokens,
    method and memory-state hash. Different memory methods never share entries."""

    def __init__(self, root: Path, backend_desc: dict):
        self.dir = root / stable_hash(backend_desc)
        self.dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.dir / "backend.json", backend_desc)
        self.store = JsonlStore(self.dir / "responses.jsonl")
        self._mem: dict[str, dict] | None = None

    @staticmethod
    def key(image_sha: str, prompt_hash: str, backend_desc: dict, max_new_tokens: int,
            method: str, memory_hash: str) -> str:
        return stable_hash([image_sha, prompt_hash, backend_desc, max_new_tokens, method, memory_hash], 32)

    def get(self, key: str) -> InferenceResult | None:
        if key not in self.store:
            return None
        if self._mem is None:
            self._mem = {r["key"]: r for r in self.store.iter()}
        r = self._mem.get(key)
        if r is None:
            self._mem = {r["key"]: r for r in self.store.iter()}
            r = self._mem.get(key)
        if r is None:
            return None
        d = dict(r["result"])
        return InferenceResult(**d)

    def put(self, key: str, res: InferenceResult) -> None:
        if res.error or res.synthetic:
            return        # never cache failures or synthetic outputs
        rec = {"key": key, "result": asdict(res)}
        if self.store.append(rec) and self._mem is not None:
            self._mem[key] = rec


class Trace:
    """Externally observable tool trace (no model 'reasoning' is treated as a trace)."""

    def __init__(self, path: Path):
        self.store = JsonlStore(path)
        self._n = len(self.store)

    def log(self, tool: str, **kw) -> None:
        self._n += 1
        self.store.append({"key": f"{self._n:09d}", "tool": tool, **kw})


def write_prompt(run_dir: Path, prompt_hash: str, system: str, user: str) -> str:
    p = run_dir / "prompts" / f"{prompt_hash}.json"
    if not p.exists():
        atomic_write_json(p, {"system": system, "user": user})
    return str(p.relative_to(run_dir))


def write_raw(run_dir: Path, key: str, res: InferenceResult) -> str:
    p = run_dir / "raw" / key[:2] / f"{key}.json"
    if not p.exists():
        atomic_write_text(p, json.dumps(asdict(res)))
    return str(p.relative_to(run_dir))
