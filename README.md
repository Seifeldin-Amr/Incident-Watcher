# OpsPilot

OpsPilot is an AI-powered incident investigation and SRE agent MVP. Phase 1 provides the backend API and durable incident storage; later phases will add diagnostics, RAG, and LLM analysis.

## Phase 1: Run locally

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Start the API:

```powershell
python -m uvicorn app.main:app --reload
```

Open Swagger UI at <http://127.0.0.1:8000/docs>.

Run tests:

```powershell
python -m pytest
```

The default SQLite database is `opspilot.db`. Set `OPSPILOT_DATABASE_PATH` to use a different location.

## Phase 1 endpoints

- `GET /health` checks that the service is running.
- `POST /api/v1/incidents` creates an incident.
- `GET /api/v1/incidents` lists recent incidents.
- `GET /api/v1/incidents/{incident_id}` retrieves one incident.

## RAG pipeline

The RAG component provides log and official documentation ingestion and
retrieval through `rag.knowledge_ingest` and `rag.knowledge_query`.

Install its optional dependencies with:

```powershell
python -m pip install -e ".[rag]"
```

## Local LLM

The local LLM  uses `google/flan-t5-small`, an approximately
80M-parameter model. It is intentionally separate from the API and loads the
model only when the generator is run. Install the optional dependencies with:

```powershell
python -m pip install -e ".[llm]"
```

Generate the RAG prompt, then pass it to the local model:

```powershell
python -m rag.knowledge_query "sshd authentication failures" --prompt-only | python -m llm.generate
```

Alternatively, provide an existing RAG prompt through a file:

```powershell
python -m llm.generate --prompt-file incident_prompt.txt
```

The first run downloads the small model from Hugging Face and caches it
locally. Backend/API integration is intentionally not included yet.

## Standalone RAG/LLM service

The standalone service keeps the embedding model and LLM loaded in one
long-running process so they are not initialized for every prompt. It exposes
separate retrieval, prompt, generation, and full-analysis endpoints.

Run the service directly with the local Python environment:

```powershell
python -m pip install -e ".[rag,llm,dev]"
python -m rag_service
```

Open <http://127.0.0.1:8010/docs>. The service provides:

- `GET /health` for process health.
- `GET /ready` for model and vector-store readiness.
- `POST /v1/retrieve` to inspect retrieved evidence.
- `POST /v1/prompt` to inspect the complete generated prompt.
- `POST /v1/generate` to test the LLM with a supplied prompt.
- `POST /v1/analyze` to run retrieval, prompt construction, and generation.

Run the RAG/LLM service in Docker when Docker Desktop is installed:

```powershell
docker compose -f docker-compose.rag-llm.yml up --build
```

The container exposes port `8010`, persists model files in the
`opspilot-model-cache` volume, and reads the existing Chroma databases from
`storage/`. Docker is intentionally not required by the Python test suite.