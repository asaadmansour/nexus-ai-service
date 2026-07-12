import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.assessment_generation import generate_assessment


router = APIRouter(
    prefix="/agents",
    tags=["Assessment Generation Agent"]
)

logger = logging.getLogger(__name__)


class GenerateAssessmentRequest(BaseModel):
    skills: List[str] = Field(default_factory=list)
    years_experience: Optional[int] = Field(
        default=None,
        alias="yearsExperience"
    )
    headline: Optional[str] = None

    question_count: int = Field(
        default=6,
        alias="questionCount",
        ge=1,
        le=20
    )

    duration_seconds: int = Field(
        default=1800,
        alias="durationSeconds",
        ge=60
    )


@router.post("/generate-assessment")
def generate_assessment_route(
    request: GenerateAssessmentRequest
):

    try:
        result = generate_assessment(
            skills=request.skills,
            years_experience=request.years_experience,
            headline=request.headline,
            question_count=request.question_count,
            duration_seconds=request.duration_seconds
        )

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as exc:
        logger.exception(
            "Assessment generation failed."
        )

        raise HTTPException(
            status_code=503,
            detail="Assessment generation is temporarily unavailable.",
        ) from exc