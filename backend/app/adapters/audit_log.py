"""Privacy-minimised SQLite audit event storage for the local demo."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class AuditEvent:
    endpoint: str
    latency_ms: int
    criteria_version: str
    model_version: str
    flag: str | None = None
    probability: float | None = None


class SQLiteAuditLog:
    """Store only operational summary fields; never raw patient-like input."""

    def __init__(self, database_path: str):
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self._path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    criteria_version TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    flag TEXT,
                    probability REAL
                )
                """
            )

    def write(self, event: AuditEvent) -> None:
        with sqlite3.connect(self._path) as connection:
            connection.execute(
                """
                INSERT INTO audit_events (
                    created_at, endpoint, latency_ms, criteria_version,
                    model_version, flag, probability
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    event.endpoint,
                    event.latency_ms,
                    event.criteria_version,
                    event.model_version,
                    event.flag,
                    event.probability,
                ),
            )

    def count(self) -> int:
        """Small test/diagnostic helper; no event payload is returned."""
        with sqlite3.connect(self._path) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0])
