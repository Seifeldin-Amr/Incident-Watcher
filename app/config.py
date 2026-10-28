from __future__ import annotations

import os
from pathlib import Path


def database_path() -> Path:
    """Return the SQLite path, allowing local deployments to override it."""
    configured_path = os.getenv("OPSPILOT_DATABASE_PATH", "opspilot.db")
    return Path(configured_path)