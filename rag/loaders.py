from __future__ import annotations

import csv
import re
from pathlib import Path

from .models import SourceDocument, SourceType


_SYSLOG_PATTERN = re.compile(
    r"^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<process>[\w-]+)(?:\[(?P<pid>\d+)\])?:\s+"
    r"(?P<message>.*)$"
)


def _log_document(
    *,
    document_id: str,
    title: str,
    message: str,
    dataset_uri: str,
    metadata: dict[str, str],
) -> SourceDocument:
    if not message.strip():
        raise ValueError(f"Empty OpenSSH log message for {document_id}")
    return SourceDocument(
        document_id=document_id,
        title=title,
        text=message,
        system="openssh",
        source_type=SourceType.HISTORICAL_LOG,
        uri=dataset_uri,
        metadata=metadata,
    )


def load_openssh_csv_logs(
    path: Path,
    *,
    dataset_uri: str,
) -> list[SourceDocument]:
    """Load the dataset's parsed CSV while retaining every row and message."""
    documents: list[SourceDocument] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"LogID", "Timestamp", "Host", "PID", "Message", "LogType"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected OpenSSH parsed CSV columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            log_id = row["LogID"].strip()
            message = row["Message"]
            if not log_id or not message.strip():
                raise ValueError(f"{path}:{row_number}: missing LogID or Message")
            documents.append(
                _log_document(
                    document_id=f"openssh-log-{log_id}-{row_number}",
                    title=f"OpenSSH log {log_id}",
                    message=message,
                    dataset_uri=dataset_uri,
                    metadata={
                        "record_id": log_id,
                        "source_file": path.name,
                        "row_number": str(row_number),
                        "timestamp": row["Timestamp"],
                        "host": row["Host"],
                        "pid": row["PID"],
                        "event_type": row["LogType"],
                    },
                )
            )
    return documents


def load_openssh_raw_logs(
    path: Path,
    *,
    dataset_uri: str,
) -> list[SourceDocument]:
    """Load raw OpenSSH syslog lines, preserving the original line as text."""
    documents: list[SourceDocument] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = _SYSLOG_PATTERN.match(line)
        metadata = {
            "record_id": str(line_number),
            "source_file": path.name,
            "row_number": str(line_number),
        }
        if match:
            metadata.update(
                {
                    key: value
                    for key, value in match.groupdict().items()
                    if value is not None
                }
            )
        documents.append(
            _log_document(
                document_id=f"openssh-raw-log-{line_number}",
                title=f"OpenSSH raw log line {line_number}",
                message=line,
                dataset_uri=dataset_uri,
                metadata=metadata,
            )
        )
    return documents