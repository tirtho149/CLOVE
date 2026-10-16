"""clove.inference.prompt — prompt templates. Exactly ONE image (the query) per call.

`clove` style: lettered options (only the r candidates) + "0 = none of these",
cited evidence lines under each option, JSON-only answer `{"answer": "<code>"}` so the
answer is one short token whose top-logprobs give a distribution over options.

`sage0` style: the SAGE open_agentic k=0 wording (vendored from
sage-wacv2027/wacv/prompts.py, refs and part-index removed because k=0), with one
added sentence allowing UNKNOWN (required for open-world scoring; logged departure).
"""
from __future__ import annotations

import string
from dataclasses import dataclass

from ..memory.policy import Selection
from ..util import stable_hash

UNKNOWN_CODE = "0"

CLOVE_SYSTEM = (
    "You identify soybean diseases from one photo. Compare the photo with the options "
    "and their cited evidence. Choose the single best matching option. If no option "
    "matches the photo, choose 0. Reply with JSON only, no explanation: "
    '{"answer": "<option code>"}'
)

SAGE0_SYSTEM = (
    "You are an expert plant pathologist classifying a diseased plant image.\n\n"
    "## Strategy\n"
    "1. Read the test image first. Note the affected plant part (leaf, stem, pod, "
    "root, whole plant) and key visual features (color, shape, pattern, texture).\n"
    "2. Review the symptom descriptions and class list below.\n"
    "3. Based ONLY on the test image and symptom descriptions, identify the "
    "best matching class. Do NOT use the Read, Glob, or any other tool to "
    "find or view additional images. Only read the test image.\n"
    "- Submit your prediction.\n\n"
    "End your response with exactly this JSON block:\n"
    "```json\n"
    '{"prediction": "<class_name>", "confidence": <0.0-1.0>, '
    '"reasoning": "<brief explanation>"}\n'
    "```\n"
    "The prediction MUST be one of the provided class names (exact match)."
    "\nIf the image matches none of the provided classes, use \"UNKNOWN\" as the prediction."
)


def option_codes(n: int) -> list[str]:
    letters = string.ascii_uppercase
    codes = list(letters[:n])
    i = 0
    while len(codes) < n:
        codes.append(letters[i // 26] + letters[i % 26])
        i += 1
    return codes


@dataclass
class BuiltPrompt:
    system: str
    user_text: str
    style: str
    code_to_class: dict[str, str]
    instruction_text: str          # everything except evidence lines and class names
    hash: str


def build_prompt(sel: Selection, display: dict[str, str] | None = None) -> BuiltPrompt:
    display = display or {}
    if sel.prompt_style == "sage0":
        return _build_sage0(sel)
    codes = option_codes(len(sel.candidates))
    lines = ["Options:"]
    for code, cid in zip(codes, sel.candidates):
        lines.append(f"{code}) {display.get(cid, cid.replace('_', ' '))}")
        for ev in sel.evidence.get(cid, []):
            lines.append(f"   {ev.text}")
    lines.append(f"{UNKNOWN_CODE}) none of these (unknown)")
    lines.append("JSON:")
    user = "\n".join(lines)
    instr = CLOVE_SYSTEM + "\nOptions:\n0) none of these (unknown)\nJSON:"
    c2c = dict(zip(codes, sel.candidates))
    return BuiltPrompt(CLOVE_SYSTEM, user, "clove", c2c, instr,
                       stable_hash([CLOVE_SYSTEM, user]))


def _build_sage0(sel: Selection) -> BuiltPrompt:
    parts = ["## Test Image\n\nThe image you need to classify is attached."]
    class_list = "\n".join(f"- {c}" for c in sorted(sel.candidates))
    parts.append(f"## Possible Classes ({len(sel.candidates)} total)\n\n{class_list}")
    kb = []
    for c in sorted(sel.candidates):
        txt = " ".join(e.text for e in sel.evidence.get(c, []))
        if txt:
            kb.append(f"### {c}\n{txt}\n")
    if kb:
        parts.append("## Symptom Descriptions (Knowledge Base)\n\n" + "\n".join(kb))
    parts.append("Now begin: Read the test image, reason through the evidence, "
                 "and end with your prediction JSON.")
    user = "\n\n".join(parts)
    c2c = {c: c for c in sel.candidates}
    return BuiltPrompt(SAGE0_SYSTEM, user, "sage0", c2c, SAGE0_SYSTEM,
                       stable_hash([SAGE0_SYSTEM, user]))
