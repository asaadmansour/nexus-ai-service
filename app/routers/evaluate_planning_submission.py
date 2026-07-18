import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.planning_submission_evaluation import (
    PlanningSubmissionEvaluationError,
    evaluate_submission,
)

router = APIRouter(prefix="/agents", tags=["Planning Submission Evaluation"])

logger = logging.getLogger(__name__)


# ── Request Models (local to router) ─────────────────────────────────────

class SubmissionInput(BaseModel):
    submissionType: str  # "architecture" or "ui_ux"
    summary: str
    content: Dict[str, Any]


class EvaluateSubmissionRequest(BaseModel):
    project: Dict[str, Any]
    brief: Dict[str, Any]
    submission: SubmissionInput


# ── Router Endpoint ──────────────────────────────────────────────────────

@router.post("/evaluate-planning-submission")
def evaluate_planning_submission_route(request: EvaluateSubmissionRequest):
    """
    Evaluate a planning submission (architecture or UI/UX) and return a recommendation,
    score, strengths, risks, and suggested admin notes.
    """
    try:
        # Pass the entire request as a dict to the agent
        result = evaluate_submission(request.dict())
        return result

    except PlanningSubmissionEvaluationError as e:
        logger.exception("Planning submission evaluation failed: %s", str(e))
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": "30"},
        ) from e

    except ValueError as e:
        logger.warning("Invalid input for submission evaluation: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as exc:
        logger.exception("Unexpected error in planning submission evaluation")
        raise HTTPException(
            status_code=503,
            detail="Submission evaluation is temporarily unavailable.",
        ) from exc