from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

if __package__:
    from .embeddings import SentenceTransformerEmbedder
    from .knowledge_base import OPENSSH_CSV_URL, load_openssh_documentation
    from .pipeline import ingest_documents
    from .vector_store import ChromaVectorStore
    from .loaders import load_openssh_csv_logs
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rag.embeddings import SentenceTransformerEmbedder
    from rag.knowledge_base import OPENSSH_CSV_URL, load_openssh_documentation
    from rag.pipeline import ingest_documents
    from rag.vector_store import ChromaVectorStore
    from rag.loaders import load_openssh_csv_logs


def _download(url: str, destination: Path) -> None:
    print(f"Downloading OpenSSH dataset to {destination}...", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "OpsPilot OpenSSH knowledge ingester"})
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


class _Spinner:
    def __init__(self, message: str) -> None:
        self.message = message
        self._running = False
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Spinner":
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        self._running = False
        if self._thread:
            self._thread.join()
        status = "failed" if exc_type else "done"
        print(f"\r{self.message} {status}.      ", flush=True)

    def _run(self) -> None:
        frames = "|/-\\"
        index = 0
        while self._running:
            print(f"\r{self.message} {frames[index % len(frames)]}", end="", flush=True)
            index += 1
            time.sleep(0.15)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build separate OpenSSH log and documentation KBs.")
    parser.add_argument(
        "--csv", type=Path, default=Path("storage/openssh_access_parsed.csv")
    )
    parser.add_argument("--logs-db-dir", type=Path, default=Path("storage/openssh_logs"))
    parser.add_argument("--docs-db-dir", type=Path, default=Path("storage/openssh_docs"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-words", type=int, default=400)
    parser.add_argument("--overlap-words", type=int, default=60)
    args = parser.parse_args()

    if not args.csv.exists():
        _download(OPENSSH_CSV_URL, args.csv)
    with _Spinner("Loading embedding model"):
        embedder = SentenceTransformerEmbedder(args.model)
    with _Spinner("Loading OpenSSH log records"):
        logs = load_openssh_csv_logs(args.csv, dataset_uri=OPENSSH_CSV_URL)
    print(f"Loaded {len(logs)} log records.", flush=True)
    with _Spinner("Loading official OpenSSH documentation"):
        docs = load_openssh_documentation()
    print(f"Loaded {len(docs)} documentation sections.", flush=True)
    with _Spinner("Embedding and writing log database"):
        log_count = ingest_documents(
            logs,
            embedder=embedder,
            vector_store=ChromaVectorStore(
                args.logs_db_dir, collection_name="openssh_logs"
            ),
            max_words=args.max_words,
            overlap_words=args.overlap_words,
        )
    with _Spinner("Embedding and writing documentation database"):
        doc_count = ingest_documents(
            docs,
            embedder=embedder,
            vector_store=ChromaVectorStore(
                args.docs_db_dir, collection_name="openssh_docs"
            ),
            max_words=args.max_words,
            overlap_words=args.overlap_words,
        )
    print(
        f"Indexed {len(logs)} log records ({log_count} chunks) and "
        f"{len(docs)} documentation sections ({doc_count} chunks)."
    )


if __name__ == "__main__":
    main()
