"""Response shape for a criteria assessment."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CriterionItem(BaseModel):
    id: str
    group: str
    label: str
    source: str
    severity: Literal["priority", "advisory"]
    input: Literal["clinician", "data"]
    note: str
    status: Literal["pass", "fail", "unknown"]


class GroupSummary(BaseModel):
    label: str
    note: str
    total: int
    pass_count: int = Field(validation_alias="pass_", serialization_alias="pass")
    fail: int
    unknown: int


class CriteriaSummary(BaseModel):
    pass_count: int = Field(validation_alias="pass_", serialization_alias="pass")
    fail: int
    unknown: int
    priority_not_met: list[str]
    priority_unknown: list[str]


class CriteriaResponse(BaseModel):
    criteria_version: str
    disclaimer: str
    summary: CriteriaSummary
    groups: dict[str, GroupSummary]
    items: list[CriterionItem]

    @classmethod
    def from_engine(cls, result: dict[str, Any]) -> "CriteriaResponse":
        """Translate engine keys named 'pass' into valid Python field aliases."""
        summary = {**result["summary"], "pass_": result["summary"]["pass"]}
        groups = {
            name: {**group, "pass_": group["pass"]}
            for name, group in result["groups"].items()
        }
        return cls(
            criteria_version=result["criteria_version"],
            disclaimer=result["disclaimer"],
            summary=summary,
            groups=groups,
            items=result["items"],
        )
