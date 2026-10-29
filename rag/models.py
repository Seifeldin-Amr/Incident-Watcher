from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SourceType(StrEnum):
    OFFICIAL_DOCUMENTATION = "official_documentation"
    RUNBOOK = "runbook"
    HISTORICAL_LOG = "historical_log"


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    title: str
    text: str
    system: str
    source_type: SourceType
    uri: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict[str, str] = field(default_factory=dict)

    def vector_metadata(self) -> dict[str, str | int]:
        return {
            **self.metadata,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    distance: float
    metadata: dict[str, str | int]