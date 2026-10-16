"""clove.contracts — typed records shared by every module.

Everything that crosses a module boundary is a frozen dataclass or an Enum defined
here, so the leakage invariant ("every claim and label belongs to the active scope")
can be checked in one place (`assert_in_scope`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"          # scored as the reject symbol ⊥
    UNRESOLVED = "UNRESOLVED"    # operational state; scored as ⊥ (pre-registered mapping)


class GTStatus(str, Enum):
    """Status of an image's ground-truth class at the episode it is evaluated."""
    OLD = "old"                          # admitted at an earlier episode
    NEW = "new"                          # admitted at this episode
    PENDING_UNKNOWN = "pending_unknown"  # eligible, admitted at a later episode
    PERMANENT_UNKNOWN = "permanent_unknown"  # never admitted (no KB / held back)


class ClaimKind(str, Enum):
    QUOTE = "quote"              # verbatim quote from a cited web source
    FIELD = "field"              # structured registry field (organ, type, pathogen) with cited quote
    KB_DESCRIPTION = "kb_description"  # SAGE KB description sentence (no per-sentence URL)
    LOOK_ALIKE = "look_alike"    # documented look-alike statement (source-cited)


class FailureCode(str, Enum):
    NONE = "NONE"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    KNOWN_FALSE_UNKNOWN = "KNOWN_FALSE_UNKNOWN"
    UNKNOWN_FALSE_KNOWN = "UNKNOWN_FALSE_KNOWN"
    CLASS_COMPETITION = "CLASS_COMPETITION"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    LOSSY_MEMORY = "LOSSY_MEMORY"
    CITATION_MISMATCH = "CITATION_MISMATCH"
    FUTURE_LEAKAGE = "FUTURE_LEAKAGE"
    OUTPUT_PARSE = "OUTPUT_PARSE"
    OOM = "OOM"
    TIMEOUT = "TIMEOUT"
    INVALID_IMAGE = "INVALID_IMAGE"
    REFERENCE_IMAGE_VIOLATION = "REFERENCE_IMAGE_VIOLATION"
    OUTPUT_CAP_EXCEEDED = "OUTPUT_CAP_EXCEEDED"
    UNRESOLVED = "UNRESOLVED"
    UNKNOWN_ROOT_CAUSE = "UNKNOWN_ROOT_CAUSE"


class RunStatus(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


class FutureClassLeak(PermissionError):
    """Raised whenever a non-admitted class reaches a prompt, index or memory."""


@dataclass(frozen=True)
class AccessScope:
    episode: int
    allowed_class_ids: frozenset[str]
    inventory_hash: str

    def check(self, class_id: str, what: str = "class") -> None:
        if class_id not in self.allowed_class_ids:
            raise FutureClassLeak(f"{what} {class_id!r} not admitted at episode {self.episode}")


@dataclass(frozen=True)
class Claim:
    claim_id: str
    class_id: str
    kind: ClaimKind
    text: str                    # text shown to the model
    verbatim_quote: str          # source quote ('' when the source gives none)
    source_url: str              # '' for KB descriptions without per-claim URL
    source_doc: str              # which SAGE file the claim came from
    field: str = ""              # registry field name, e.g. visual_symptoms.summary
    mentions: tuple[str, ...] = ()   # other canonical class ids named in the text


@dataclass(frozen=True)
class ClassRecord:
    class_id: str
    display_name: str
    aliases: tuple[str, ...]
    disease_type: str
    affected_parts: tuple[str, ...]
    pathogen: str
    look_alikes: tuple[str, ...]     # canonical ids of documented look-alikes


@dataclass(frozen=True)
class ImageRecord:
    image_id: str            # content sha1 (SAGE pool id) or sha256 when built from files
    path: str
    class_id: str
    split: str               # 'test' | 'dev'
    source_dataset: str
    group_id: str            # specimen / near-duplicate cluster
    sha256: str
    variant_id: str = "orig"
    shift_type: str = "iid"
    severity: float = 0.0
    rights_status: str = "sage-cc-by-4.0-unverified"


@dataclass(frozen=True)
class EvidenceLine:
    claim_id: str
    class_id: str
    text: str
    tokens: int


@dataclass
class InferenceResult:
    text: str
    input_tokens: int                 # exact backend count incl. image tokens when exposed
    output_tokens: int
    latency_s: float
    image_tokens: int | None = None   # None => backend does not expose it
    option_logprobs: dict[str, float] | None = None   # top logprobs at the answer position
    token_logprobs: list[dict[str, float]] | None = None
    error: str | None = None
    synthetic: bool = False           # True only for MockBackend outputs
    cost_usd: float | None = None     # reported by hosted backends (claude headless)
    meta: dict[str, Any] | None = None  # backend-specific accounting (turns, cache tokens, tools)


@dataclass
class Prediction:
    """One immutable per-image prediction record (the unit of all analysis)."""
    key: str
    run_id: str
    order_id: str
    seed: int
    episode: int
    method: str
    budget_id: str
    image_id: str
    variant_id: str
    shift_type: str
    severity: float
    group_id: str
    source_dataset: str
    gt_class: str
    gt_status: str
    gt_admit_episode: int | None
    decision: str
    label_id: str | None
    raw_answer: str | None
    unknown_score: float | None
    known_scores: dict[str, float]
    candidates: list[str]
    evidence_ids: list[str]
    kb_tokens: int
    prompt_tokens: int
    output_tokens: int
    image_tokens: int | None
    vlm_calls: int
    latency_s: float
    failure: str
    class_set_hash: str
    memory_state_hash: str
    prompt_hash: str
    inference_config_hash: str
    gate_threshold: float | None
    raw_response_path: str
    cached: bool = False
    synthetic: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        if self.gt_status in (GTStatus.OLD.value, GTStatus.NEW.value):
            return self.decision == Decision.KNOWN.value and self.label_id == self.gt_class
        return self.decision != Decision.KNOWN.value

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["correct"] = self.correct
        return d


def assert_in_scope(scope: AccessScope, claims, label_id: str | None) -> None:
    """Leakage invariant from protocol §11."""
    for c in claims:
        scope.check(c.class_id, "claim class")
        for m in c.mentions:
            scope.check(m, f"class mentioned in claim {c.claim_id}")
    if label_id is not None:
        scope.check(label_id, "predicted label")
