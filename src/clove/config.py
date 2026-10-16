"""clove.config — YAML configuration with defaults, path resolution and hashing.

`configs/pilot.yaml` is the single entry point; it may `include:` budgets.yaml and
stats.yaml. Relative paths are resolved against the directory of the config file
that declares them, so the package runs from any working directory.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .util import stable_hash

DEFAULTS: dict[str, Any] = {
    "run_name": "soybean_pilot",
    "crop": "soybean",
    "paths": {
        "wacv_manifest": None,
        "kb_xlsx": None,
        "provenance_json": None,
        "registry_json": None,
        "prepared_dataset": None,
        "image_dirs": [],
        "image_meta_csv": None,
        "hard_pairs_csv": None,
        "artifacts": "artifacts",
    },
    "classes": {
        "aliases": {},
        "exclude": [],
        "hold_out": [],
        "min_test_images": 10,
        "min_claims": 1,
        "unify_case": True,
    },
    "dev": {"use_manifest_refs": True, "max_per_class": 8},
    "dedup": {"near_dup_hamming": 6},
    "schedule": {"initial": 10, "batch": 5, "stages": 4, "min_initial": 4, "min_batch": 2},
    "orders": {"random_orders": 3, "confusable": True, "reversed": True, "seed": 2026,
               "hard_pair_source": "registry_look_alikes"},
    "model": {
        "backend": "openai_compat",
        "model_id": "Qwen/Qwen3-VL-32B-Instruct",
        "revision": None,
        "base_url": "http://localhost:8000",
        "api_key_env": None,
        "tokenizer": "Qwen/Qwen3-VL-32B-Instruct",
        "require_exact_tokenizer": True,
        "temperature": 0.0,
        "top_logprobs": 10,
        "timeout_s": 180,
        "max_retries": 2,
        "claude": {"bin": "claude", "model": "claude-sonnet-5-5", "image_mode": "inline",
                   "effort": "low", "strip_api_key": True, "enforce_output_cap": True,
                   "extra_args": []},
    },
    "retrieval": {"visual": "none", "siglip_model": "google/siglip-so400m-patch14-384",
                  "precomputed_scores": None, "device": "auto"},
    "gate": {"mode": "threshold", "target_known_tpr": 0.95, "calibrate": "per_episode",
             "dev_per_class": 5, "unresolved_maps_to": "UNKNOWN"},
    "policy": {"second_call": False, "second_call_band": [0.35, 0.65], "max_calls": 2},
    "methods": ["names", "sage0", "append", "fresh", "prism"],
    "memory": {"card_tokens": 128, "mask_unadmitted_mentions": True},
    "prism": {"alpha": 1.0, "beta": 0.5, "gamma": 0.25, "refresh_top_k": 3,
              "interference_threshold": 0.15},
    "budgets": {
        "primary": "b512_r4_o96",
        "grid": {"b512_r4_o96": {"kb_tokens": 512, "r": 4, "max_new_tokens": 96,
                                 "instruction_tokens": 180, "excerpt_tokens": 96}},
        # per-method b_out exceptions (logged departure from matched caps), e.g. {sage0: 512}
        "method_max_new_tokens": {},
    },
    "stats": {
        "bootstrap": 10000, "ci": 0.95, "seed": 12345,
        "noninferiority_margin": 0.02,
        "primary": {"metric": "old_class_macro_accuracy", "treatment": "prism",
                    "control": "append", "episodes": "post_acquisition"},
        "primary_family": [["prism", "append"]],
    },
    "compute": {"n_gpus": 1, "max_gpu_hours": 24.0, "max_usd": 50.0, "max_calls": 60000, "parallel": 1,
                "checkpoint_every": 25},
    "shifts": {"enabled": [], "covariate": {}, "prevalence": [0.10, 0.25, 0.50]},
    "smoke": {"classes": 3, "images_per_class": 3, "dev_per_class": 2},
}

PATH_KEYS = ("wacv_manifest", "kb_xlsx", "provenance_json", "registry_json",
             "prepared_dataset", "image_meta_csv", "hard_pairs_csv", "artifacts")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "grid":
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(p: str | None, base: Path) -> str | None:
    if p is None or p == "":
        return None
    q = Path(p).expanduser()
    return str(q if q.is_absolute() else (base / q).resolve())


def _load_with_includes(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    merged: dict = {}
    for inc in raw.pop("include", []) or []:
        merged = _merge(merged, _load_with_includes((path.parent / inc).resolve()))
    # resolve this file's own paths relative to its directory
    if "paths" in raw:
        for k in PATH_KEYS:
            if k in raw["paths"]:
                raw["paths"][k] = _resolve(raw["paths"][k], path.parent)
        if "image_dirs" in raw["paths"]:
            raw["paths"]["image_dirs"] = [_resolve(d, path.parent) for d in raw["paths"]["image_dirs"]]
    return _merge(merged, raw)


@dataclass(frozen=True)
class Budget:
    budget_id: str
    kb_tokens: int
    r: int
    max_new_tokens: int
    instruction_tokens: int
    excerpt_tokens: int


class Config:
    def __init__(self, data: dict, source: str | None = None):
        self.data = data
        self.source = source

    @classmethod
    def load(cls, path: str | Path, overrides: dict | None = None) -> "Config":
        path = Path(path).resolve()
        data = _merge(DEFAULTS, _load_with_includes(path))
        if overrides:
            data = _merge(data, overrides)
        if not Path(data["paths"]["artifacts"]).is_absolute():
            data["paths"]["artifacts"] = _resolve(data["paths"]["artifacts"], path.parent)
        cfg = cls(data, str(path))
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        cfg = cls(_merge(DEFAULTS, data))
        cfg.validate()
        return cfg

    def __getitem__(self, k: str) -> Any:
        return self.data[k]

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    @property
    def artifacts(self) -> Path:
        return Path(self.data["paths"]["artifacts"])

    def budget(self, budget_id: str | None = None) -> Budget:
        bid = budget_id or self.data["budgets"]["primary"]
        b = self.data["budgets"]["grid"][bid]
        return Budget(bid, int(b["kb_tokens"]), int(b["r"]), int(b["max_new_tokens"]),
                      int(b.get("instruction_tokens", 180)), int(b.get("excerpt_tokens", 96)))

    def budgets(self) -> list[Budget]:
        return [self.budget(b) for b in self.data["budgets"]["grid"]]

    def hash(self) -> str:
        return stable_hash(self.data)

    def validate(self) -> None:
        s = self.data["schedule"]
        if s["initial"] < 1 or s["batch"] < 1 or s["stages"] < 1:
            raise ValueError("schedule values must be positive")
        if self.data["budgets"]["primary"] not in self.data["budgets"]["grid"]:
            raise ValueError("budgets.primary must name a budgets.grid entry")
        known = {"names", "sage0", "append", "fresh", "prism", "uniform", "recency",
                 "random", "bm25", "oracle", "shuffled", "prism_noscan", "prism_noprotect"}
        bad = [m for m in self.data["methods"] if m not in known]
        if bad:
            raise ValueError(f"unknown methods: {bad}")
        if self.data["gate"]["unresolved_maps_to"] not in ("UNKNOWN",):
            raise ValueError("pre-registered mapping: UNRESOLVED must be scored as UNKNOWN")
