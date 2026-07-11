from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.routers.shared_models import ProjectSpec

router = APIRouter(prefix="/agents", tags=["Evaluation Agent"])


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
    criteria = request.task.acceptance_criteria or [
        "Submission addresses the requested task.",
        "Submission includes enough evidence for review.",
    ]

    rubric = [
        {
            "criterion": criterion,
            "met": index == 0,
            "evidence": (
                "Mock evidence found in submitted artifact."
                if index == 0
                else "Mock review says this item still needs verification."
            ),
        }
        for index, criterion in enumerate(criteria[:5])
    ]

    return {
        "passed": False,
        "score": 2,
        "revisionRequested": True,
        "revisionNotes": (
            "Mock evaluation: the submission has a useful starting point, but one or "
            "more acceptance criteria still need stronger evidence."
        ),
        "requiresHumanReview": request.submission.submission_type in {"figma", "other"},
        "rubric": rubric,
    }
