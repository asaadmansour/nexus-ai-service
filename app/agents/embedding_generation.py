import logging
from typing import List

from google import genai
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_DIMENSIONS = 1024
MAX_TEXT_LENGTH = 8000


class EmbeddingResponse(BaseModel):
    embedding: List[float] = Field(default_factory=list)
    model: str
    dimensions: int


def generate_embedding(
    text: str,
    dimensions: int = DEFAULT_DIMENSIONS,
    model: str | None = None,
) -> dict:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Text is required for embedding generation.")

    if dimensions <= 0:
        raise ValueError("Embedding dimensions must be greater than zero.")

    if dimensions > 4096:
        raise ValueError("Embedding dimensions cannot exceed 4096.")

    embedding_model = (model or DEFAULT_EMBEDDING_MODEL).strip()
    if not embedding_model:
        raise ValueError("Embedding model cannot be empty.")

    source_text = cleaned_text[:MAX_TEXT_LENGTH]
    client = genai.Client()

    try:
        response = client.models.embed_content(
            model=embedding_model,
            contents=source_text,
            config={
                "output_dimensionality": dimensions,
                "task_type": "RETRIEVAL_DOCUMENT",
            },
        )

        embeddings = response.embeddings or []
        if not embeddings or not getattr(embeddings[0], "values", None):
            raise ValueError("Embedding model returned no values.")

        values = [float(value) for value in embeddings[0].values]
        if len(values) != dimensions:
            raise ValueError(
                f"Embedding model returned {len(values)} dimensions instead of {dimensions}."
            )

        return EmbeddingResponse(
            embedding=values,
            model=embedding_model,
            dimensions=len(values),
        ).model_dump()

    except ValueError:
        raise

    except Exception as exc:
        logger.exception("Embedding generation failed.")
        raise ValueError("Failed to generate embedding using AI.") from exc
