import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.embedding_generation import generate_embedding


router = APIRouter(
    prefix="/agents",
    tags=["Embedding Agent"]
)

logger = logging.getLogger(__name__)


class GenerateEmbeddingRequest(BaseModel):
    text: str = Field(min_length=1)
    dimensions: int = Field(default=1024, ge=128, le=4096)
    model: str | None = None


@router.post("/generate-embedding")
def generate_embedding_route(request: GenerateEmbeddingRequest):
    try:
        return generate_embedding(
            text=request.text,
            dimensions=request.dimensions,
            model=request.model,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Embedding generation failed.")
        raise HTTPException(
            status_code=503,
            detail="Embedding generation is temporarily unavailable.",
        ) from exc
