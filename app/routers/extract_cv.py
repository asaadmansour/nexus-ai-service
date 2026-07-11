import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.cv_extraction import process_cv_with_llm

router = APIRouter(
    prefix="/agents",
    tags=["CV Extraction Agent"]
)

logger = logging.getLogger(__name__)


class ExtractCvRequest(BaseModel):
    cv_url: str = Field(
        alias="cvUrl",
        min_length=5
    )


@router.post("/extract-cv")
def extract_cv(request: ExtractCvRequest):

    try:
        result = process_cv_with_llm(request.cv_url)

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as exc:
        logger.exception("CV extraction failed.")

        raise HTTPException(
            status_code=503,
            detail="CV extraction is temporarily unavailable.",
        ) from exc