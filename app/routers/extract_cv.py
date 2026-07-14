import logging

from fastapi import APIRouter, HTTPException
from pydantic import AnyHttpUrl, BaseModel, Field

from app.agents.cv_extraction import (
    CVExtractionServiceError,
    process_cv_with_llm,
)

router = APIRouter(
    prefix="/agents",
    tags=["CV Extraction Agent"]
)

logger = logging.getLogger(__name__)


class ExtractCvRequest(BaseModel):
    cv_url: AnyHttpUrl = Field(
        alias="cvUrl"
    )


@router.post("/extract-cv")
def extract_cv(request: ExtractCvRequest):

    try:
        result = process_cv_with_llm(str(request.cv_url))

        return result

    except CVExtractionServiceError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={
                "Retry-After": "30"
            },
        ) from e

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        ) from e

    except Exception as exc:
        logger.exception("CV extraction failed.")

        raise HTTPException(
            status_code=503,
            detail="CV extraction is temporarily unavailable.",
        ) from exc
