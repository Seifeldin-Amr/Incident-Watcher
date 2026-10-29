from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .embeddings import SentenceTransformerEmbedder
from .vector_store import ChromaVectorStore


def build_where(system: str | None, source_type: str | None) -> dict[str, Any] | None:
    conditions = []
    if system:
        conditions.append({"system": system})
    if source_type:
        conditions.append({"source_type": source_type})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local OpsPilot Chroma knowledge base.")
    parser.add_argument("query")
    parser.add_argument("--db-dir", type=Path, default=Path("storage/vector_db"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--source-type")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder(args.model)
    store = ChromaVectorStore(args.db_dir)
    results = store.query(
        embedder.embed_query(args.query),
        n_results=args.top_k,
        where=build_where(args.system, args.source_type),
    )
    print(
        json.dumps(
            [
                {
                    "chunk_id": result.chunk_id,
                    "distance": result.distance,
                    "text": result.text,
                    "metadata": result.metadata,
                }
                for result in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()