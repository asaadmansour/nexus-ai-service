from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.routers.shared_models import ProjectSpec

router = APIRouter(prefix="/agents", tags=["Scrum Master Agent"])


class ProjectForTaskGeneration(BaseModel):
    project_id: str = Field(alias="projectId")
    title: str
    budget_min: float = Field(alias="budgetMin")
    budget_max: float = Field(alias="budgetMax")
    currency: str
    deadline_date: str | None = Field(default=None, alias="deadlineDate")
    status: str | None = None


class BriefForTaskGeneration(BaseModel):
    brief_id: str = Field(alias="briefId")
    summary: str | None = None
    project_type: str | None = Field(default=None, alias="projectType")
    domain: str | None = None
    technical: dict[str, Any] | None = None
    non_functional: dict[str, Any] | None = Field(default=None, alias="nonFunctional")
    deliverables: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list, alias="requiredSkills")
    preferred_skills: list[str] = Field(default_factory=list, alias="preferredSkills")
    acceptance_criteria: list[str] = Field(default_factory=list, alias="acceptanceCriteria")
    experience_level: str | None = Field(default=None, alias="experienceLevel")
    experience_min_years: int | None = Field(default=None, alias="experienceMinYears")
    ai_decided: dict[str, Any] | None = Field(default=None, alias="aiDecided")


class GenerateTaskRequest(BaseModel):
    task_type: Literal["phase_0_spec", "implementation_breakdown"] = Field(alias="taskType")
    project: ProjectForTaskGeneration
    brief: BriefForTaskGeneration
    project_spec: ProjectSpec | None = Field(default=None, alias="projectSpec")


@router.post("/generate-task")
def generate_task(request: GenerateTaskRequest):
    if request.task_type == "phase_0_spec":
        return generate_phase_0_spec_task(request.project, request.brief)

    if request.project_spec is None:
        raise HTTPException(
            status_code=400,
            detail="projectSpec is required for implementation_breakdown.",
        )

    return generate_implementation_breakdown(
        request.project,
        request.brief,
        request.project_spec,
    )


def generate_phase_0_spec_task(
    project: ProjectForTaskGeneration,
    brief: BriefForTaskGeneration,
):
    return {
        "taskType": "phase_0_spec",
        "tasks": [
            {
                "title": "Architecture & UI/UX Specification",
                "description": (
                    "Create the locked technical and product specification for "
                    f"{project.title}. The spec must translate the brief into API "
                    "contracts, UI flows, design tokens, implementation conventions, "
                    "and acceptance criteria for later task evaluation."
                ),
                "taskOrder": 1,
                "isSpecTask": True,
                "requiredSkills": [
                    "System Design",
                    "API Design",
                    "UI/UX",
                    *brief.required_skills,
                ],
                "preferredSkills": brief.preferred_skills,
                "estimatedHours": 12,
                "allocatedBudget": round(project.budget_max * 0.15, 2),
                "deliverables": [
                    "API contract",
                    "Core user flows",
                    "Design tokens",
                    "Folder and naming conventions",
                    "Evaluation rubric for future tasks",
                ],
                "acceptanceCriteria": [
                    "API contract covers the core project entities and flows.",
                    "UI/UX flows cover the main customer and freelancer journeys.",
                    "Design tokens define colors, spacing, typography, and states.",
                    "Implementation conventions are clear enough for future freelancers.",
                    "Evaluation criteria are specific enough to judge later submissions.",
                ],
                "searchCriteria": (
                    "Find a freelancer strong in system design, API contracts, "
                    "UI/UX planning, and writing implementation specifications."
                ),
            }
        ],
    }


def generate_implementation_breakdown(
    project: ProjectForTaskGeneration,
    brief: BriefForTaskGeneration,
    project_spec: ProjectSpec,
):
    return {
        "taskType": "implementation_breakdown",
        "milestones": [
            {
                "title": "Foundation",
                "description": "Set up the base application using the locked project spec.",
                "milestoneOrder": 1,
            }
        ],
        "tasks": [
            {
                "title": f"Build core {brief.domain or 'project'} foundation",
                "description": (
                    "Implement the first production-ready slice using the locked "
                    "architecture, UI conventions, and API contract."
                ),
                "taskOrder": 2,
                "isSpecTask": False,
                "requiredSkills": brief.required_skills,
                "preferredSkills": brief.preferred_skills,
                "estimatedHours": 40,
                "allocatedBudget": round(project.budget_max * 0.25, 2),
                "acceptanceCriteria": brief.acceptance_criteria,
                "searchCriteria": (
                    "Find a freelancer who matches the required skills and can follow "
                    f"the locked spec status: {project_spec.status or 'locked'}."
                ),
            }
        ],
    }
