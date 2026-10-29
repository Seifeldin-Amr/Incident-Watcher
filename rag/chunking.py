from __future__ import annotations

import hashlib

from .models import DocumentChunk, SourceDocument


def _stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{document_id}-{chunk_index:04d}-{digest}"


def chunk_document(
    document: SourceDocument,
    max_words: int = 400,
    overlap_words: int = 60,
) -> list[DocumentChunk]:
    if max_words < 1:
        raise ValueError("max_words must be positive")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between zero and max_words - 1")

    words = document.text.split()
    if not words:
        return []

    chunks = []
    start = 0
    chunk_index = 0
    step = max_words - overlap_words
    while start < len(words):
        chunk_words = words[start : start + max_words]
        chunk_text = " ".join(chunk_words)
        chunks.append(
            DocumentChunk(
                chunk_id=_stable_chunk_id(document.document_id, chunk_index, chunk_text),
                document_id=document.document_id,
                text=chunk_text,
                chunk_index=chunk_index,
                metadata={
                    "title": document.title,
                    "system": document.system,
                    "source_type": document.source_type.value,
                    "uri": document.uri,
                    **document.metadata,
                },
            )
        )
        start += step
        chunk_index += 1
    return chunks