"""Map separate criteria and ML signals into a review prompt, never a decision."""
from __future__ import annotations

from enum import Enum


class FlagCode(str, Enum):
    NONE = "NONE"
    ML_HIGH_CRITERIA_MET = "ML_HIGH_CRITERIA_MET"
    CRITERIA_OPEN_ML_LOW = "CRITERIA_OPEN_ML_LOW"
    BOTH_REVIEW = "BOTH_REVIEW"


_MESSAGES = {
    FlagCode.NONE: None,
    FlagCode.ML_HIGH_CRITERIA_MET: (
        "ML estimates higher facility-care likelihood although listed priority criteria are met. "
        "Review the contributing factors and clinical context."
    ),
    FlagCode.CRITERIA_OPEN_ML_LOW: (
        "Priority criteria are not met or are unknown although the ML estimate is lower. "
        "Complete and review the checklist."
    ),
    FlagCode.BOTH_REVIEW: (
        "Both the checklist and ML estimate indicate that closer review is needed."
    ),
}


def disagreement(priority_open: int, is_high: bool) -> tuple[FlagCode, str | None]:
    """Return a review prompt from checklist completeness and ML category."""
    if priority_open == 0 and not is_high:
        code = FlagCode.NONE
    elif priority_open == 0 and is_high:
        code = FlagCode.ML_HIGH_CRITERIA_MET
    elif priority_open > 0 and not is_high:
        code = FlagCode.CRITERIA_OPEN_ML_LOW
    else:
        code = FlagCode.BOTH_REVIEW
    return code, _MESSAGES[code]
