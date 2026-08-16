import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.planning_submission_evaluation import (
    PlanningSubmissionEvaluationError,
    evaluate_submission,
)

router = APIRouter(prefix="/agents", tags=["Planning Submission Evaluation"])
logger = logging.getLogger(__name__)


class PlanningRequirement(BaseModel):
    key: str
    title: str
    description: str
    mandatory: bool = True
    requiresUrl: bool = False
    applicability: Literal["required", "optional"] = "required"
    allowNotApplicable: bool = False
    rationale: str = ""


class SubmissionInput(BaseModel):
    submissionId: str
    submissionVersion: int = Field(ge=1)
    submissionType: Literal["architecture", "ui_ux"]
    title: Optional[str] = None
    summary: Optional[str] = None
    content: Dict[str, Any] = Field(default_factory=dict)
    fileUrls: Dict[str, Any] = Field(default_factory=dict)


class EvaluateSubmissionRequest(BaseModel):
    project: Dict[str, Any]
    brief: Dict[str, Any] = Field(default_factory=dict)
    requirements: List[PlanningRequirement]
    requirementProfile: Dict[str, Any] = Field(default_factory=dict)
    submission: SubmissionInput
    approvedArchitecture: Optional[Dict[str, Any]] = None
    previousVerdict: Optional[Dict[str, Any]] = None


@router.post("/evaluate-planning-submission")
def evaluate_planning_submission_route(request: EvaluateSubmissionRequest):
    """Evaluate the applicable project-scaled requirements and return revision work."""
    try:
        return evaluate_submission(request.model_dump())
    except PlanningSubmissionEvaluationError as exc:
        logger.exception("Planning submission evaluation failed: %s", str(exc))
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "30"},
        ) from exc
    except ValueError as exc:
        logger.warning("Invalid planning evaluation input: %s", str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected planning submission evaluation error")
        raise HTTPException(
            status_code=503,
            detail="Submission evaluation is temporarily unavailable.",
        ) from exc
