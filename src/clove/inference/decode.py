"""clove.inference.decode — strict output parsing and option-score extraction."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from ..contracts import InferenceResult
from .prompt import UNKNOWN_CODE, BuiltPrompt

_ANSWER = re.compile(r'"answer"\s*:\s*"([^"]{1,4})"')
_PRED = re.compile(r'"prediction"\s*:\s*"([^"]+)"')


@dataclass
class Decoded:
    ok: bool
    raw_answer: str | None
    label_id: str | None          # admitted class id, or None for unknown / parse failure
    chose_unknown: bool
    probs: dict[str, float]       # class_id -> normalised probability over options
    p_unknown: float | None
    score_source: str             # logprobs | hard | none


def _norm_tok(t: str) -> str:
    return t.strip().strip('"').strip("'").strip()


def option_distribution(res: InferenceResult, codes: list[str]) -> dict[str, float] | None:
    """Softmax over option codes using the top-logprobs at the answer position."""
    if not res.token_logprobs:
        return None
    acc = ""
    for pos in res.token_logprobs:
        tok = pos.get("__token__", "")
        if re.search(r'"answer"\s*:\s*"?\s*$', acc):
            lp: dict[str, float] = {}
            for t, v in pos.items():
                if t == "__token__":
                    continue
                k = _norm_tok(t)
                if k in codes and (k not in lp or v > lp[k]):
                    lp[k] = v
            if lp:
                m = max(lp.values())
                z = sum(math.exp(v - m) for v in lp.values())
                return {k: math.exp(lp[k] - m) / z if k in lp else 0.0 for k in codes}
        acc += tok
    return None


def decode(res: InferenceResult, prompt: BuiltPrompt) -> Decoded:
    if prompt.style == "sage0":
        return _decode_sage0(res, prompt)
    m = _ANSWER.findall(res.text or "")
    raw = m[-1].strip() if m else None
    codes = list(prompt.code_to_class) + [UNKNOWN_CODE]
    dist = option_distribution(res, codes)
    if raw is None or (raw not in codes):
        return Decoded(False, raw, None, False, {}, None, "none")
    if dist is None:
        dist = {k: 1.0 if k == raw else 0.0 for k in codes}
        src = "hard"
    else:
        src = "logprobs"
    probs = {prompt.code_to_class[k]: v for k, v in dist.items() if k in prompt.code_to_class}
    return Decoded(True, raw, None if raw == UNKNOWN_CODE else prompt.code_to_class[raw],
                   raw == UNKNOWN_CODE, probs, dist.get(UNKNOWN_CODE, 0.0), src)


def _decode_sage0(res: InferenceResult, prompt: BuiltPrompt) -> Decoded:
    text = res.text or ""
    raw = None
    for blk in reversed(re.findall(r"\{[^{}]*\"prediction\"[^{}]*\}", text)):
        try:
            raw = str(json.loads(blk)["prediction"])
            break
        except Exception:  # noqa: BLE001
            continue
    if raw is None:
        mm = _PRED.findall(text)
        raw = mm[-1] if mm else None
    if raw is None:
        return Decoded(False, None, None, False, {}, None, "none")
    lut = {c.lower(): c for c in prompt.code_to_class}
    if raw.strip().upper() == "UNKNOWN":
        return Decoded(True, raw, None, True, {}, 1.0, "hard")
    cid = lut.get(raw.strip().lower()) or lut.get(raw.strip().replace(" ", "_").lower())
    if cid is None:
        return Decoded(False, raw, None, False, {}, None, "none")
    # score: probability of the first token of the predicted value (approximation,
    # documented; SAGE output is free-form so no full option distribution exists)
    p_first = None
    if res.token_logprobs:
        acc = ""
        for pos in res.token_logprobs:
            if re.search(r'"prediction"\s*:\s*"\s*$', acc):
                tok = pos.get("__token__", "")
                if tok in pos:
                    p_first = math.exp(pos[tok])
                break
            acc += pos.get("__token__", "")
    probs = {cid: p_first if p_first is not None else 1.0}
    return Decoded(True, raw, cid, False, probs, None, "logprobs" if p_first is not None else "hard")
