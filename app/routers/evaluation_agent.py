import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.submission_evaluation import (
    SubmissionEvaluationError,
    evaluate_submission as evaluate_submission_agent,
)
from app.routers.shared_models import ProjectSpec

router = APIRouter(prefix="/agents", tags=["Evaluation Agent"])

logger = logging.getLogger(__name__)


class ProjectForEvaluation(BaseModel):
    project_id: str = Field(alias="projectId")
    title: str | None = None


class BriefForEvaluation(BaseModel):
    brief_id: str | None = Field(default=None, alias="briefId")
    summary: str | None = None
    project_type: str | None = Field(default=None, alias="projectType")
    domain: str | None = None
    acceptance_criteria: list[str] = Field(default_factory=list, alias="acceptanceCriteria")


class TaskForEvaluation(BaseModel):
    task_id: str = Field(alias="taskId")
    title: str
    description: str
    is_spec_task: bool = Field(default=False, alias="isSpecTask")
    deliverables: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list, alias="acceptanceCriteria")


class SubmissionForEvaluation(BaseModel):
    submission_id: str = Field(alias="submissionId")
    submission_type: Literal["pdf", "repo", "pull_request", "figma", "zip", "text", "other"] = Field(
        default="other",
        alias="submissionType",
    )
    submission_url: str | None = Field(default=None, alias="submissionUrl")
    repository_url: str | None = Field(default=None, alias="repositoryUrl")
    pull_request_url: str | None = Field(default=None, alias="pullRequestUrl")
    commit_sha: str | None = Field(default=None, alias="commitSha")
    submission_text: str | None = Field(default=None, alias="submissionText")
    notes: str | None = None


class EvaluateSubmissionRequest(BaseModel):
    project: ProjectForEvaluation
    task: TaskForEvaluation
    submission: SubmissionForEvaluation
    brief: BriefForEvaluation | None = None
    project_spec: ProjectSpec | None = Field(default=None, alias="projectSpec")


@router.post("/evaluate-submission")
def evaluate_submission(request: EvaluateSubmissionRequest):
    try:
        return evaluate_submission_agent(request.model_dump(by_alias=True))

    except SubmissionEvaluationError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
            headers={"Retry-After": "30"},
        ) from e

    except Exception as exc:
        logger.exception("Submission evaluation failed.")
        raise HTTPException(
            status_code=503,
            detail="Submission evaluation is temporarily unavailable.",
        ) from exc
