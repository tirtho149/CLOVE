"""clove.util — hashing, atomic IO, append-only JSONL and status files."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator

from .contracts import RunStatus


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def _default(o: Any) -> Any:
    if hasattr(o, "value"):
        return o.value
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def stable_hash(obj: Any, n: int = 16) -> str:
    return sha256_bytes(stable_json(obj).encode())[:n]


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_default) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


class JsonlStore:
    """Append-only JSONL keyed by a stable `key` field.

    Collision policy (deterministic): the FIRST record written for a key wins; later
    writes with the same key are ignored and counted, never overwrite.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys: set[str] = set()
        self.collisions = 0
        if self.path.exists():
            for rec in self.iter():
                self._keys.add(rec["key"])

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def append(self, rec: dict[str, Any]) -> bool:
        k = rec["key"]
        if k in self._keys:
            self.collisions += 1
            return False
        line = json.dumps(rec, sort_keys=True, default=_default)
        with open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._keys.add(k)
        return True

    def iter(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # a torn final line from a crash: ignore, it will be recomputed
                    continue


def write_status(out_dir: str | Path, command: str, status: RunStatus,
                 detail: dict[str, Any] | None = None,
                 resume: str | None = None) -> Path:
    p = Path(out_dir) / "status.json"
    atomic_write_json(p, {
        "command": command,
        "status": status.value,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "detail": detail or {},
        "resume_command": resume,
    })
    return p


def environment_report() -> dict[str, Any]:
    rep: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {},
    }
    for mod in ("numpy", "yaml", "PIL", "torch", "transformers", "matplotlib"):
        try:
            m = __import__(mod)
            rep["packages"][mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            rep["packages"][mod] = None
    try:
        import torch  # type: ignore
        rep["cuda_available"] = bool(torch.cuda.is_available())
        rep["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        rep["cuda_available"] = False
        rep["gpus"] = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        rep["nvidia_smi"] = out.stdout.strip().splitlines() if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        rep["nvidia_smi"] = None
    return rep


def chunked(it: Iterable, n: int) -> Iterator[list]:
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
