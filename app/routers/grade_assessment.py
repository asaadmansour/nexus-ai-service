import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.assessment_grading import grade_assessment


router = APIRouter(
    prefix="/agents",
    tags=["Assessment Grading Agent"]
)

logger = logging.getLogger(__name__)


class RubricInput(BaseModel):
    maxScore: float
    gradingNotes: str


class QuestionInput(BaseModel):
    id: str
    questionType: str
    skill: str
    difficulty: str
    prompt: str
    rubric: RubricInput


class AnswerValue(BaseModel):
    value: Optional[str] = None


class AnswerInput(BaseModel):
    questionId: str
    answer: AnswerValue


class GradeAssessmentRequest(BaseModel):
    assessmentId: str

    questions: List[QuestionInput] = Field(
        default_factory=list
    )

    answers: List[AnswerInput] = Field(
        default_factory=list
    )


@router.post("/grade-assessment")
def grade_assessment_route(
    request: GradeAssessmentRequest
):

    try:

        result = grade_assessment(
            assessment_id=request.assessmentId,
            questions=request.questions,
            answers=request.answers
        )

        return result


    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


    except Exception as exc:

        logger.exception(
            "Assessment grading failed."
        )

        raise HTTPException(
            status_code=503,
            detail="Assessment grading is temporarily unavailable.",
        ) from exc