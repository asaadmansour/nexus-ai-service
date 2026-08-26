import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.requirements.document_extraction import (
    extract_requirements_document,
)


router = APIRouter(prefix="/agents", tags=["Requirements Agent"])
logger = logging.getLogger(__name__)


class ExtractRequirementsDocumentRequest(BaseModel):
    file_name: str = Field(alias="fileName", min_length=1, max_length=255)
    mime_type: str = Field(alias="mimeType", min_length=1, max_length=150)
    content_base64: str = Field(alias="contentBase64", min_length=1)
    current_brief: dict[str, Any] = Field(default_factory=dict, alias="currentBrief")


@router.post("/extract-requirements-document")
def extract_document(request: ExtractRequirementsDocumentRequest):
    try:
        return extract_requirements_document(
            file_name=request.file_name,
            mime_type=request.mime_type,
            content_base64=request.content_base64,
            current_brief=request.current_brief,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Requirements document extraction failed.")
        raise HTTPException(
            status_code=503,
            detail="Requirements document extraction is temporarily unavailable.",
        ) from exc
