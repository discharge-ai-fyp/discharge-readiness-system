"""Use-case service that combines independent criteria and ML views."""
from __future__ import annotations

import time

from app.adapters.audit_log import AuditEvent, SQLiteAuditLog
from app.domain.disagreement import disagreement
from app.schemas.review import CaseInput, DisagreementFlag, ReviewResponse
from app.services.criteria_service import CriteriaService
from app.services.model_service import DISCLAIMER, ModelService


class ReviewService:
    """Coordinate existing services without creating a readiness score."""

    def __init__(
        self,
        criteria_service: CriteriaService,
        model_service: ModelService,
        audit_log: SQLiteAuditLog,
    ):
        self._criteria = criteria_service
        self._model = model_service
        self._audit_log = audit_log

    def review(self, case: CaseInput) -> ReviewResponse:
        started = time.perf_counter()
        criteria = self._criteria.assess(case.clinician_inputs)
        prediction = self._model.predict(case.model_features)
        priority_open = len(criteria.summary.priority_not_met) + len(
            criteria.summary.priority_unknown
        )
        code, message = disagreement(priority_open, prediction.is_high)
        response = ReviewResponse(
            criteria=criteria,
            ml=prediction,
            flag=DisagreementFlag(code=code, message=message),
            disclaimer=DISCLAIMER,
        )
        self._audit_log.write(
            AuditEvent(
                endpoint="review",
                latency_ms=int((time.perf_counter() - started) * 1000),
                criteria_version=criteria.criteria_version,
                model_version=prediction.model_version,
                flag=code.value,
                probability=prediction.probability,
            )
        )
        return response
