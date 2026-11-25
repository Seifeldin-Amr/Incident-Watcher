from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("rag_service.main:app", host="0.0.0.0", port=8010)
