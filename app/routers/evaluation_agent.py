import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.submission_evaluation import (
    SubmissionEvaluationError,
    evaluate_submission as evaluate_submission_agent,
)

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
    description: str | None = None
    is_spec_task: bool = Field(default=False, alias="isSpecTask")
    deliverables: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list, alias="acceptanceCriteria")
    integration_checks: list[str] = Field(default_factory=list, alias="integrationChecks")
    contract_references: list[str] = Field(default_factory=list, alias="contractReferences")
    owned_paths: list[str] = Field(default_factory=list, alias="ownedPaths")
    quality_criteria: list[str] = Field(default_factory=list, alias="qualityCriteria")


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
    inspection: dict[str, Any] | None = None


class PriorEvaluation(BaseModel):
    evaluation_run_id: str = Field(alias="evaluationRunId")
    submission_id: str | None = Field(default=None, alias="submissionId")
    commit_sha: str | None = Field(default=None, alias="commitSha")
    score: str | None = None
    recommendation: str | None = None
    summary: str | None = None
    unmet_criteria: list[str] = Field(default_factory=list, alias="unmetCriteria")
    completed_at: str | None = Field(default=None, alias="completedAt")


class EvaluateSubmissionRequest(BaseModel):
    project: ProjectForEvaluation
    task: TaskForEvaluation
    submission: SubmissionForEvaluation
    brief: BriefForEvaluation | None = None
    # Loose dict so the full spec (architecture/designSystem/dataModel/...) reaches
    # the prompt instead of being stripped to a fixed shared-model shape.
    project_spec: dict[str, Any] | None = Field(default=None, alias="projectSpec")
    evaluation_history: list[PriorEvaluation] = Field(
        default_factory=list,
        alias="evaluationHistory",
    )


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
