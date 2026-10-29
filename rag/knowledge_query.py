from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import SentenceTransformerEmbedder
from .knowledge_base import (
    OpenSSHIncidentRetriever,
    build_incident_prompt,
    format_incident_context,
)
from .vector_store import ChromaVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate an OpenSSH incident.")
    parser.add_argument("query")
    parser.add_argument("--logs-db-dir", type=Path, default=Path("storage/openssh_logs"))
    parser.add_argument("--docs-db-dir", type=Path, default=Path("storage/openssh_docs"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Print the complete evidence-grounded LLM prompt instead of JSON context.",
    )
    args = parser.parse_args()

    retriever = OpenSSHIncidentRetriever(
        SentenceTransformerEmbedder(args.model),
        ChromaVectorStore(args.logs_db_dir, collection_name="openssh_logs"),
        ChromaVectorStore(args.docs_db_dir, collection_name="openssh_docs"),
    )
    logs, docs = retriever.retrieve(args.query, log_k=args.top_k, documentation_k=args.top_k)
    if args.prompt_only:
        print(build_incident_prompt(args.query, logs, docs))
    else:
        print(json.dumps({"context": format_incident_context(logs, docs)}, indent=2))


if __name__ == "__main__":
    main()
