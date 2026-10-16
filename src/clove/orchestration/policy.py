"""clove.orchestration.policy — the calibrated unknown gate (tool `gate_unknown`)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts import Decision
from ..inference.decode import Decoded


def unknown_score(d: Decoded) -> float | None:
    """Higher = more likely unknown. 1 - max known-class probability (the option
    distribution includes the explicit '0' option). None if no usable score."""
    if not d.ok:
        return None
    if d.chose_unknown and not d.probs:
        return 1.0
    if not d.probs:
        return None
    return 1.0 - max(d.probs.values())


@dataclass
class UnknownGate:
    mode: str = "threshold"          # threshold | argmax_only
    target_tpr: float = 0.95
    threshold: float | None = None
    n_calibration: int = 0

    def calibrate(self, known_dev_scores: list[float]) -> float | None:
        """τ = smallest value with >= target_tpr of known dev scores <= τ. Uses ONLY
        development images of admitted classes; test images are never consulted."""
        xs = sorted(s for s in known_dev_scores if s is not None and not math.isnan(s))
        self.n_calibration = len(xs)
        if self.mode != "threshold" or not xs:
            self.threshold = None
            return None
        k = max(0, min(len(xs) - 1, math.ceil(self.target_tpr * len(xs)) - 1))
        self.threshold = xs[k]
        return self.threshold

    def decide(self, d: Decoded) -> tuple[Decision, str | None, float | None]:
        s = unknown_score(d)
        if not d.ok:
            return Decision.UNRESOLVED, None, s
        if d.chose_unknown or d.label_id is None:
            return Decision.UNKNOWN, None, s
        if self.threshold is not None and s is not None and s > self.threshold:
            return Decision.UNKNOWN, None, s
        return Decision.KNOWN, d.label_id, s


def scored_decision(dec: Decision) -> str:
    """Pre-registered mapping: UNRESOLVED is scored as the reject symbol."""
    return Decision.UNKNOWN.value if dec == Decision.UNRESOLVED else dec.value
