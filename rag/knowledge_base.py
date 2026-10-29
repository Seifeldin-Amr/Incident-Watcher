from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from urllib.request import Request, urlopen

from .loaders import load_openssh_csv_logs
from .models import RetrievedChunk, SourceDocument, SourceType
from .pipeline import Embedder, VectorStore, ingest_documents

OPENSSH_DATASET_URL = (
    "https://github.com/Rootly-AI-Labs/logs-dataset/tree/main/openssh"
)
OPENSSH_CSV_URL = (
    "https://raw.githubusercontent.com/Rootly-AI-Labs/logs-dataset/main/"
    "openssh/openssh_access_parsed.csv"
)
OPENSSH_MANUAL_URLS = {
    "sshd": "https://man.openbsd.org/sshd",
    "sshd_config": "https://man.openbsd.org/sshd_config",
}


def load_openssh_log_knowledge_base(
    csv_path: Path,
    *,
    dataset_uri: str = OPENSSH_DATASET_URL,
) -> list[SourceDocument]:
    return load_openssh_csv_logs(csv_path, dataset_uri=dataset_uri)


def _manual_sections(html: str, *, name: str, source_url: str) -> list[SourceDocument]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as error:
        raise RuntimeError(
            "Documentation ingestion requires beautifulsoup4. "
            'Install with: python -m pip install -e ".[rag]"'
        ) from error

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else name
    version_match = re.search(r"\bOpenBSD\s+([0-9][\w.-]*)", soup.get_text(" ", strip=True))
    version = version_match.group(1) if version_match else ""
    sections: list[SourceDocument] = []
    for index, heading in enumerate(soup.find_all(re.compile(r"^h[1-6]$"))):
        heading_text = heading.get_text(" ", strip=True)
        if not heading_text:
            continue
        parts = [heading_text]
        for element in heading.find_all_next():
            if element is heading:
                continue
            if element.name and re.fullmatch(r"h[1-6]", element.name):
                break
            if element.name in {"p", "pre", "dl", "ul", "ol", "table"}:
                text = element.get_text(" ", strip=True)
                if text and (not parts or text != parts[-1]):
                    parts.append(text)
        text = "\n\n".join(parts)
        if len(text.split()) < 4:
            continue
        metadata = {
            "source_url": source_url,
            "manual": name,
            "section_title": heading_text,
        }
        if version:
            metadata["version"] = version
        sections.append(
            SourceDocument(
                document_id=f"openssh-doc-{name}-{index}",
                title=f"{title}: {heading_text}",
                text=text,
                system="openssh",
                source_type=SourceType.OFFICIAL_DOCUMENTATION,
                uri=source_url,
                metadata=metadata,
            )
        )
    if not sections:
        raise ValueError(f"No meaningful sections found in OpenSSH manual {source_url}")
    return sections


def load_openssh_documentation(
    manuals: Sequence[tuple[str, str]] | None = None,
    *,
    fetcher: Callable[[str], str] | None = None,
) -> list[SourceDocument]:
    selected = manuals or tuple(OPENSSH_MANUAL_URLS.items())
    get_text = fetcher or _fetch_url
    return [
        section
        for name, url in selected
        for section in _manual_sections(get_text(url), name=name, source_url=url)
    ]


def _fetch_url(url: str) -> str:
    request = Request(url, headers={"User-Agent": "OpsPilot OpenSSH knowledge ingester"})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


class OpenSSHIncidentRetriever:
    """Retrieves log evidence first, then documentation informed by that evidence."""

    def __init__(self, embedder: Embedder, logs: VectorStore, documentation: VectorStore):
        self.embedder = embedder
        self.logs = logs
        self.documentation = documentation

    def retrieve(
        self,
        query: str,
        *,
        log_k: int = 5,
        documentation_k: int = 5,
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        log_filter = {
            "$and": [
                {"system": "openssh"},
                {"source_type": SourceType.HISTORICAL_LOG.value},
            ]
        }
        log_results = self.logs.query(
            self.embedder.embed_query(query),
            n_results=log_k,
            where=log_filter,
        )
        query_keyword = getattr(self.logs, "query_keyword", None)
        if callable(query_keyword):
            exact_results = query_keyword(
                query,
                n_results=log_k,
                where=log_filter,
            )
            seen = {result.chunk_id for result in exact_results}
            log_results = exact_results + [
                result for result in log_results if result.chunk_id not in seen
            ]
            log_results = log_results[:log_k]
        clues = " ".join(result.text for result in log_results)
        documentation_query = f"{query}\nRelevant log evidence:\n{clues}" if clues else query
        documentation_results = self.documentation.query(
            self.embedder.embed_query(documentation_query),
            n_results=documentation_k,
            where={
                "$and": [
                    {"system": "openssh"},
                    {"source_type": SourceType.OFFICIAL_DOCUMENTATION.value},
                ]
            },
        )
        return log_results, documentation_results


def format_incident_context(
    log_results: Sequence[RetrievedChunk],
    documentation_results: Sequence[RetrievedChunk],
) -> str:
    def render(results: Sequence[RetrievedChunk]) -> str:
        if not results:
            return "No relevant evidence found."
        rendered = []
        for result in results:
            source_parts = [
                str(result.metadata.get(key))
                for key in ("uri", "source_file", "record_id")
                if result.metadata.get(key)
            ]
            rendered.append(f"[{' '.join(source_parts) or 'unknown source'}] {result.text}")
        return "\n\n".join(rendered)

    return (
        "LOG EVIDENCE\n"
        f"{render(log_results)}\n\n"
        "OPENSSH DOCUMENTATION\n"
        f"{render(documentation_results)}"
    )


def build_incident_prompt(
    query: str,
    log_results: Sequence[RetrievedChunk],
    documentation_results: Sequence[RetrievedChunk],
) -> str:
    """Build the evidence-grounded prompt passed to the incident-analysis LLM."""
    return (
        "You are an incident investigator specializing in OpenSSH.\n"
        "Analyze the user's incident using only the supplied context.\n"
        "Historical logs are examples of past events, not proof of the cause of "
        "the current incident. Do not invent facts, root causes, configuration "
        "values, commands, or citations.\n\n"
        "Required response structure:\n"
        "1. Summary: briefly describe the reported problem.\n"
        "2. Confirmed observations: cite relevant log evidence and distinguish "
        "observed facts from interpretation.\n"
        "3. Possible causes: list plausible causes only when supported by the "
        "logs or official documentation; label each as unconfirmed.\n"
        "4. Documentation support: explain which official OpenSSH behavior or "
        "configuration applies, with its source URL.\n"
        "5. Verification steps: provide safe, non-destructive checks. Do not "
        "recommend deleting keys, changing access controls, restarting services, "
        "or other disruptive actions unless justified and explicitly confirmed.\n"
        "6. Evidence gaps: state clearly what cannot be concluded.\n"
        "7. Sources: list the source URL and record ID or filename for every "
        "material claim.\n\n"
        "Use 'not confirmed' when the context is insufficient. Do not treat a "
        "similar historical event as confirmation of the current root cause.\n\n"
        f"USER QUERY\n{query}\n\n"
        f"{format_incident_context(log_results, documentation_results)}"
    )


def generate_openssh_incident_report(
    query: str,
    *,
    retriever: OpenSSHIncidentRetriever,
    llm: Callable[[str], str],
    log_k: int = 5,
    documentation_k: int = 5,
) -> str:
    """Retrieve context and pass an evidence-grounded prompt to the existing LLM."""
    logs, documentation = retriever.retrieve(
        query, log_k=log_k, documentation_k=documentation_k
    )
    return llm(build_incident_prompt(query, logs, documentation))


def ingest_openssh_knowledge(
    *,
    csv_path: Path,
    embedder: Embedder,
    log_store: VectorStore,
    documentation_store: VectorStore,
    max_words: int = 400,
    overlap_words: int = 60,
) -> tuple[int, int]:
    logs = load_openssh_log_knowledge_base(csv_path)
    docs = load_openssh_documentation()
    return (
        ingest_documents(
            logs,
            embedder=embedder,
            vector_store=log_store,
            max_words=max_words,
            overlap_words=overlap_words,
        ),
        ingest_documents(
            docs,
            embedder=embedder,
            vector_store=documentation_store,
            max_words=max_words,
            overlap_words=overlap_words,
        ),
    )
