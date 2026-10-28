from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import IncidentCreate


class IncidentRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    service TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    occurred_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, incident: IncidentCreate) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        occurred_at = incident.occurred_at.isoformat() if incident.occurred_at else None
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO incidents
                    (title, description, service, severity, occurred_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident.title,
                    incident.description,
                    incident.service,
                    incident.severity.value,
                    occurred_at,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            incident_id = cursor.lastrowid
        return self.get_by_id(incident_id)

    def get_by_id(self, incident_id: int) -> dict[str, Any] | None:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM incidents WHERE id = ?", (incident_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, limit: int) -> list[dict[str, Any]]:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for field_name in ("occurred_at", "created_at", "updated_at"):
            if result[field_name] is not None:
                result[field_name] = datetime.fromisoformat(result[field_name])
        return result