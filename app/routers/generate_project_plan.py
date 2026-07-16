import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.project_plan_generation import (
    ProjectPlanGenerationError,
    generate_project_plan,
    ProjectPlanRequest,
)

router = APIRouter(prefix="/agents", tags=["Project Planning"])

logger = logging.getLogger(__name__)


# ── Request Models (mirroring handoff schemas) ─────────────────────────────

class ProjectInfo(BaseModel):
    id: str
    title: str
    description: str
    status: str
    budgetMin: float
    budgetMax: float
    currency: str
    deadline: Optional[str] = None
    isDeadlineFlexible: bool = False


class BriefInfo(BaseModel):
    projectType: Optional[str] = None
    businessDomain: Optional[str] = None
    mainGoal: Optional[str] = None
    targetUsers: Optional[str] = None
    coreFeatures: List[str] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=list)
    deliverables: List[str] = Field(default_factory=list)
    constraintsPreferences: List[str] = Field(default_factory=list)
    clientBackground: Optional[str] = None
    rawBrief: Dict[str, Any] = Field(default_factory=dict)


class SubmissionInfo(BaseModel):
    id: str
    summary: str
    content: Dict[str, Any]


class PlanningTeamMember(BaseModel):
    roleKey: str
    freelancerProfileId: str
    headline: Optional[str] = None


class GeneratePlanRequest(BaseModel):
    projectPlanJobId: str
    project: ProjectInfo
    brief: BriefInfo
    architectureSubmission: SubmissionInfo
    uiuxSubmission: SubmissionInfo
    planningTeam: List[PlanningTeamMember]


# ── Router Endpoint ─────────────────────────────────────────────────────────

@router.post("/generate-project-plan")
def generate_project_plan_route(request: GeneratePlanRequest):
    """
    Generate a complete project plan (milestones, tasks, dependencies, etc.)
    from the project brief, approved architecture, and UI/UX submissions.
    """
    try:
        # Convert request to the agent's input model
        input_data = ProjectPlanRequest(
            projectPlanJobId=request.projectPlanJobId,
            project=request.project.dict(),
            brief=request.brief.dict(),
            architectureSubmission=request.architectureSubmission.dict(),
            uiuxSubmission=request.uiuxSubmission.dict(),
            planningTeam=[m.dict() for m in request.planningTeam],
        )

        result = generate_project_plan(input_data)
        return result

    except ProjectPlanGenerationError as e:
        # AI provider or validation error – 503 or 400
        logger.exception("Project plan generation failed: %s", str(e))
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": "30"},
        ) from e

    except ValueError as e:
        logger.warning("Invalid input for project plan generation: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as exc:
        logger.exception("Unexpected error in project plan generation")
        raise HTTPException(
            status_code=503,
            detail="Project plan generation is temporarily unavailable.",
        ) from exc