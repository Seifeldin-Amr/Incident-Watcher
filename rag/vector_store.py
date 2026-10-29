from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import DocumentChunk, RetrievedChunk


class ChromaVectorStore:
    """Persistent Chroma collection storing chunk text, embeddings, and citations."""

    _MAX_UPSERT_BATCH_SIZE = 5_000

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "sre_knowledge",
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as error:
            raise RuntimeError(
                "RAG dependencies are missing. Install with: "
                "python -m pip install -e \".[rag]\""
            ) from error
        persist_directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = client.get_or_create_collection(name=collection_name)

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Every document chunk must have exactly one embedding")
        if not chunks:
            return
        for start in range(0, len(chunks), self._MAX_UPSERT_BATCH_SIZE):
            batch_chunks = chunks[start : start + self._MAX_UPSERT_BATCH_SIZE]
            batch_embeddings = embeddings[start : start + self._MAX_UPSERT_BATCH_SIZE]
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch_chunks],
                documents=[chunk.text for chunk in batch_chunks],
                embeddings=batch_embeddings,
                metadatas=[chunk.vector_metadata() for chunk in batch_chunks],
            )

    def query(
        self,
        embedding: list[float],
        *,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if n_results < 1:
            raise ValueError("n_results must be positive")
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
        )
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=documents[index],
                distance=float(distances[index]),
                metadata=metadatas[index] or {},
            )
            for index, chunk_id in enumerate(ids)
        ]

    def query_keyword(
        self,
        phrase: str,
        *,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if not phrase.strip():
            return []
        results = self.collection.get(
            where=where,
            where_document={"$contains": phrase},
            limit=n_results,
            include=["documents", "metadatas"],
        )
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=documents[index],
                distance=0.0,
                metadata=metadatas[index] or {},
            )
            for index, chunk_id in enumerate(ids)
        ]

    def count(self) -> int:
        return int(self.collection.count())