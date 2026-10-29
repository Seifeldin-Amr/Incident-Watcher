from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .chunking import chunk_document
from .models import DocumentChunk, SourceDocument


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None: ...


def ingest_documents(
    documents: Sequence[SourceDocument],
    *,
    embedder: Embedder,
    vector_store: VectorStore,
    max_words: int = 400,
    overlap_words: int = 60,
) -> int:
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, max_words=max_words, overlap_words=overlap_words)
    ]
    if not chunks:
        return 0
    embeddings = embedder.embed_documents([chunk.text for chunk in chunks])
    vector_store.upsert(chunks, embeddings)
    return len(chunks)