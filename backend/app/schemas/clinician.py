"""Validated inputs used by the criteria checklist."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClinicianInputs(BaseModel):
    """Optional fields preserve unknown status when information is unavailable."""

    model_config = ConfigDict(extra="forbid")

    temperature_c: float | None = Field(default=None, ge=30, le=45)
    heart_rate: int | None = Field(default=None, ge=20, le=250)
    resp_rate: int | None = Field(default=None, ge=4, le=80)
    systolic_bp: int | None = Field(default=None, ge=40, le=300)
    spo2: float | None = Field(default=None, ge=50, le=100)
    oral_intake_ok: bool | None = None
    mental_status_normal: bool | None = None
    a1c_result: Literal[">8", ">7", "Norm", "None", "Not_measured"] | None = None
    discharge_plan_documented: bool | None = None
    med_reconciliation_done: bool | None = None
    followup_scheduled: bool | None = None
    pending_tests_plan: bool | None = None
    services_equipment_arranged: bool | None = None
