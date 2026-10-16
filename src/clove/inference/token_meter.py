"""clove.inference.token_meter — tokenizer-exact accounting (tool `budget_meter`).

Three budgets are kept apart everywhere:
  b_KB   retrieved KB evidence tokens (counted here, exact, before the call)
  b_out  max generation tokens (enforced by the backend, counted from its output)
  r      number of candidate classes offered
and k_img, which is identically 0. Prompt tokens (incl. image tokens) come from the
backend's own count after the call; they are never inferred from `max_new_tokens`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


class Tokenizer(Protocol):
    exact: bool
    name: str

    def count(self, text: str) -> int: ...


class HFTokenizer:
    exact = True

    def __init__(self, name_or_path: str, revision: str | None = None):
        from transformers import AutoTokenizer  # optional dependency
        self.tok = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
        self.name = f"hf:{name_or_path}@{revision or 'main'}"

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))


class RegexTokenizer:
    """Deterministic approximation (words + punctuation, long words split every 4
    chars). Used for CPU tests and planning only; `exact=False` is recorded in every
    record so it can never be mistaken for real accounting."""
    exact = False
    name = "regex-approx"
    _pat = re.compile(r"[A-Za-z]+|\d|[^\sA-Za-z\d]")

    def count(self, text: str) -> int:
        n = 0
        for m in self._pat.finditer(text):
            w = m.group(0)
            n += 1 + (len(w) - 1) // 4 if w.isalpha() else 1
        return n


def load_tokenizer(model_cfg: dict, allow_approx: bool = False) -> Tokenizer:
    """Claude's tokenizer is not public: with ``tokenizer: approx`` b_KB is counted by the
    regex approximation (flagged ``tokenizer_exact=false`` in every record) while prompt
    and output tokens still come exactly from the API usage."""
    if model_cfg.get("tokenizer") == "approx":
        if model_cfg.get("require_exact_tokenizer", True) and not allow_approx:
            raise RuntimeError("tokenizer=approx requires model.require_exact_tokenizer=false")
        return RegexTokenizer()
    try:
        return HFTokenizer(model_cfg["tokenizer"], model_cfg.get("revision"))
    except Exception as e:  # noqa: BLE001
        if model_cfg.get("require_exact_tokenizer", True) and not allow_approx:
            raise RuntimeError(f"exact tokenizer {model_cfg['tokenizer']!r} unavailable ({e}); "
                               "set model.require_exact_tokenizer=false only for dry runs") from e
        return RegexTokenizer()


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetMeter:
    tokenizer: Tokenizer
    log: list[dict] = field(default_factory=list)

    def count(self, text: str) -> int:
        return self.tokenizer.count(text)

    def check(self, component: str, text: str, cap: int) -> int:
        n = self.count(text)
        self.log.append({"component": component, "tokens": n, "cap": cap})
        if n > cap:
            raise BudgetExceeded(f"{component}: {n} tokens > cap {cap}")
        return n
