from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from llm.inference import DEFAULT_MODEL, LocalIncidentLLM
from rag.embeddings import SentenceTransformerEmbedder
from rag.knowledge_base import (
    OpenSSHIncidentRetriever,
    build_incident_prompt,
)
from rag.vector_store import ChromaVectorStore

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LOGS_DB_DIR = Path("storage/openssh_logs")
DEFAULT_DOCS_DB_DIR = Path("storage/openssh_docs")


class AnalysisRequest(BaseModel):
    query: str = Field(min_length=1)
    log_k: int = Field(default=5, ge=1, le=50)
    documentation_k: int = Field(default=5, ge=1, le=50)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)


class PromptRequest(BaseModel):
    query: str = Field(min_length=1)
    log_k: int = Field(default=5, ge=1, le=50)
    documentation_k: int = Field(default=5, ge=1, le=50)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "distance": chunk.distance,
        "metadata": chunk.metadata,
    }


class RagLlmService:
    def __init__(
        self,
        retriever: OpenSSHIncidentRetriever,
        llm: LocalIncidentLLM,
        *,
        embedding_model: str,
        llm_model: str,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.embedding_model = embedding_model
        self.llm_model = llm_model

    def retrieve(
        self, query: str, *, log_k: int, documentation_k: int
    ) -> tuple[list[Any], list[Any]]:
        return self.retriever.retrieve(
            query, log_k=log_k, documentation_k=documentation_k
        )

    def build_prompt(
        self, query: str, *, log_k: int, documentation_k: int
    ) -> tuple[str, list[Any], list[Any]]:
        logs, documentation = self.retrieve(
            query, log_k=log_k, documentation_k=documentation_k
        )
        return build_incident_prompt(query, logs, documentation), logs, documentation


def _setting(name: str, default: str) -> str:
    return os.getenv(name, default)


def build_service() -> RagLlmService:
    embedding_model = _setting(
        "OPSPILOT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )
    llm_model = _setting("OPSPILOT_LLM_MODEL", DEFAULT_MODEL)
    logs_db_dir = Path(
        _setting("OPSPILOT_LOGS_DB_DIR", str(DEFAULT_LOGS_DB_DIR))
    )
    docs_db_dir = Path(
        _setting("OPSPILOT_DOCS_DB_DIR", str(DEFAULT_DOCS_DB_DIR))
    )

    embedder = SentenceTransformerEmbedder(embedding_model)
    retriever = OpenSSHIncidentRetriever(
        embedder,
        ChromaVectorStore(logs_db_dir, collection_name="openssh_logs"),
        ChromaVectorStore(docs_db_dir, collection_name="openssh_docs"),
    )
    llm = LocalIncidentLLM(llm_model)
    return RagLlmService(
        retriever,
        llm,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )


def create_app(
    service: RagLlmService | None = None,
    *,
    load_service_on_startup: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if service is not None:
            application.state.rag_llm = service
        elif load_service_on_startup:
            application.state.rag_llm = build_service()
        yield

    application = FastAPI(
        title="OpsPilot RAG and LLM Service",
        description="Local retrieval-augmented generation service.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready", tags=["system"])
    def ready(request: Request) -> dict[str, Any]:
        loaded_service = getattr(request.app.state, "rag_llm", None)
        if loaded_service is None:
            return {"status": "starting"}
        return {
            "status": "ready",
            "embedding_model_loaded": True,
            "llm_loaded": True,
            "vector_stores_ready": True,
            "embedding_model": loaded_service.embedding_model,
            "llm_model": loaded_service.llm_model,
        }

    def get_service(request: Request) -> RagLlmService:
        loaded_service = getattr(request.app.state, "rag_llm", None)
        if loaded_service is None:
            raise HTTPException(status_code=503, detail="RAG/LLM service is not ready")
        return loaded_service

    @application.post("/v1/retrieve", tags=["rag"])
    def retrieve(payload: PromptRequest, request: Request) -> dict[str, Any]:
        loaded_service = get_service(request)
        logs, documentation = loaded_service.retrieve(
            payload.query,
            log_k=payload.log_k,
            documentation_k=payload.documentation_k,
        )
        return {
            "query": payload.query,
            "logs": [_chunk_to_dict(chunk) for chunk in logs],
            "documentation": [_chunk_to_dict(chunk) for chunk in documentation],
        }

    @application.post("/v1/prompt", tags=["rag"])
    def prompt(payload: PromptRequest, request: Request) -> dict[str, Any]:
        loaded_service = get_service(request)
        built_prompt, logs, documentation = loaded_service.build_prompt(
            payload.query,
            log_k=payload.log_k,
            documentation_k=payload.documentation_k,
        )
        return {
            "query": payload.query,
            "prompt": built_prompt,
            "logs_used": len(logs),
            "documentation_used": len(documentation),
        }

    @application.post("/v1/generate", tags=["llm"])
    def generate(payload: GenerateRequest, request: Request) -> dict[str, str]:
        loaded_service = get_service(request)
        return {
            "text": loaded_service.llm.generate(
                payload.prompt, max_new_tokens=payload.max_new_tokens
            )
        }

    @application.post("/v1/analyze", tags=["rag", "llm"])
    def analyze(payload: AnalysisRequest, request: Request) -> dict[str, Any]:
        loaded_service = get_service(request)
        built_prompt, logs, documentation = loaded_service.build_prompt(
            payload.query,
            log_k=payload.log_k,
            documentation_k=payload.documentation_k,
        )
        return {
            "query": payload.query,
            "answer": loaded_service.llm.generate(
                built_prompt, max_new_tokens=payload.max_new_tokens
            ),
            "prompt": built_prompt,
            "logs": [_chunk_to_dict(chunk) for chunk in logs],
            "documentation": [_chunk_to_dict(chunk) for chunk in documentation],
        }

    return application


app = create_app()
