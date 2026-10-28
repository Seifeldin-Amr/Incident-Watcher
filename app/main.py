from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from .config import database_path as configured_database_path
from .database import IncidentRepository
from .logging_config import configure_logging
from .schemas import IncidentCreate, IncidentListResponse, IncidentResponse

logger = logging.getLogger("opspilot.api")


def create_app(database_path: Path | None = None) -> FastAPI:
    repository = IncidentRepository(database_path or configured_database_path())

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        configure_logging()
        repository.initialize()
        logger.info("application_started", extra={"event": "application_started"})
        yield
        logger.info("application_stopped", extra={"event": "application_stopped"})

    application = FastAPI(
        title="OpsPilot API",
        description="API foundation for AI-powered incident investigation.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @application.get("/health", tags=["system"])
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    @application.post(
        "/api/v1/incidents",
        response_model=IncidentResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["incidents"],
    )
    def create_incident(incident: IncidentCreate) -> IncidentResponse:
        created_incident = repository.create(incident)
        logger.info(
            "incident_created",
            extra={"event": "incident_created", "incident_id": created_incident["id"]},
        )
        return IncidentResponse.model_validate(created_incident)

    @application.get(
        "/api/v1/incidents",
        response_model=IncidentListResponse,
        tags=["incidents"],
    )
    def list_incidents(
        limit: int = Query(default=50, ge=1, le=100),
    ) -> IncidentListResponse:
        incidents = repository.list(limit)
        return IncidentListResponse(
            items=[IncidentResponse.model_validate(item) for item in incidents],
            total=len(incidents),
        )

    @application.get(
        "/api/v1/incidents/{incident_id}",
        response_model=IncidentResponse,
        tags=["incidents"],
    )
    def get_incident(incident_id: int) -> IncidentResponse:
        incident = repository.get_by_id(incident_id)
        if incident is None:
            raise HTTPException(status_code=404, detail="Incident not found")
        return IncidentResponse.model_validate(incident)

    return application


app = create_app()