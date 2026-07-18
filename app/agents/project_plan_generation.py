import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, validator
from google import genai
from google.genai import errors, types

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
GENAI_TIMEOUT = 120.0


class ProjectPlanGenerationError(RuntimeError):
    """Raised when the AI provider or validation fails."""


# ── Response Models ─────────────────────────────────────────────────────

class TimelinePhase(BaseModel):
    name: str
    durationDays: int
    description: str


class Timeline(BaseModel):
    estimatedWeeks: int
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    phases: List[TimelinePhase] = Field(default_factory=list)


class Milestone(BaseModel):
    clientKey: str
    title: str
    description: str
    orderIndex: int
    estimatedDays: int
    budgetAmount: float
    currency: str
    acceptanceCriteria: List[str] = Field(default_factory=list)


class Task(BaseModel):
    clientKey: str
    milestoneClientKey: str
    title: str
    description: str
    priority: str
    roleKey: str
    requiredSkills: List[str] = Field(default_factory=list)
    estimatedHours: int
    orderIndex: int
    acceptanceCriteria: List[str] = Field(default_factory=list)
    status: Optional[str] = "todo"

    @validator('priority')
    def validate_priority(cls, v):
        allowed = {"low", "medium", "high", "urgent"}
        if v not in allowed:
            raise ValueError(f"Priority must be one of {allowed}")
        return v

    @validator('status')
    def validate_status(cls, v):
        allowed = {"todo", "blocked", "in_progress", "review", "changes_requested", "done", "cancelled"}
        if v and v not in allowed:
            raise ValueError(f"Status must be one of {allowed}")
        return v


class Dependency(BaseModel):
    taskClientKey: str
    dependsOnTaskClientKey: str
    dependencyType: str = "blocks"
    notes: Optional[str] = None

    @validator('dependencyType')
    def validate_dependency_type(cls, v):
        allowed = {"blocks", "related", "after"}
        if v not in allowed:
            raise ValueError(f"dependencyType must be one of {allowed}")
        return v


class RecommendedRole(BaseModel):
    roleKey: str
    count: int
    skills: List[str] = Field(default_factory=list)


class TeamPlan(BaseModel):
    recommendedRoles: List[RecommendedRole] = Field(default_factory=list)
    suggestedTeamSize: int


class Risk(BaseModel):
    risk: str
    impact: str
    mitigation: str


class ProjectSpec(BaseModel):
    architecture: Dict[str, Any] = Field(default_factory=dict)
    designSystem: Dict[str, Any] = Field(default_factory=dict)
    apiContract: Dict[str, Any] = Field(default_factory=dict)
    dataModel: Dict[str, Any] = Field(default_factory=dict)
    conventions: Dict[str, Any] = Field(default_factory=dict)


class ProjectPlanResponse(BaseModel):
    summary: str
    assumptions: List[str] = Field(default_factory=list)
    timeline: Timeline
    milestones: List[Milestone] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    dependencies: List[Dependency] = Field(default_factory=list)
    teamPlan: TeamPlan
    riskRegister: List[Risk] = Field(default_factory=list)
    projectSpec: ProjectSpec


# ── Input Model ──────────────────────────────────────────────────────────

class ProjectPlanRequest(BaseModel):
    projectPlanJobId: str
    project: Dict[str, Any]
    brief: Dict[str, Any]
    architectureSubmission: Dict[str, Any]
    uiuxSubmission: Dict[str, Any]
    planningTeam: List[Dict[str, Any]]


# ── Validation Helper ──────────────────────────────────────────────────

def validate_and_normalize_plan(data: Dict[str, Any]) -> ProjectPlanResponse:
    try:
        plan = ProjectPlanResponse(**data)
    except ValidationError as e:
        raise ProjectPlanGenerationError(f"Response validation failed: {e}")

    milestone_keys = {m.clientKey for m in plan.milestones}
    task_keys = {t.clientKey for t in plan.tasks}

    if len(milestone_keys) != len(plan.milestones):
        raise ProjectPlanGenerationError("Duplicate milestone clientKey found.")
    if len(task_keys) != len(plan.tasks):
        raise ProjectPlanGenerationError("Duplicate task clientKey found.")

    for task in plan.tasks:
        if task.milestoneClientKey not in milestone_keys:
            raise ProjectPlanGenerationError(
                f"Task '{task.clientKey}' references non-existent milestone '{task.milestoneClientKey}'"
            )

    all_task_keys = task_keys
    for dep in plan.dependencies:
        if dep.taskClientKey not in all_task_keys:
            raise ProjectPlanGenerationError(
                f"Dependency references unknown task '{dep.taskClientKey}'"
            )
        if dep.dependsOnTaskClientKey not in all_task_keys:
            raise ProjectPlanGenerationError(
                f"Dependency references unknown task '{dep.dependsOnTaskClientKey}'"
            )

    # Detect cycles
    graph = {key: [] for key in all_task_keys}
    for dep in plan.dependencies:
        graph[dep.taskClientKey].append(dep.dependsOnTaskClientKey)

    visited = set()
    rec_stack = set()

    def has_cycle(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    for node in all_task_keys:
        if node not in visited:
            if has_cycle(node):
                raise ProjectPlanGenerationError(
                    f"Circular dependency detected involving task '{node}'"
                )

    return plan


# ── Main Agent Function ────────────────────────────────────────────────

def generate_project_plan(request: ProjectPlanRequest) -> Dict[str, Any]:
    prompt = _build_prompt(request)
    client = genai.Client()

    try:
        response = _generate_plan_response(client, prompt)
        if not response.text:
            raise ProjectPlanGenerationError("Empty response from AI.")
        result = json.loads(response.text)
        validated_plan = validate_and_normalize_plan(result)
        return validated_plan.dict()

    except ProjectPlanGenerationError:
        raise
    except errors.APIError as e:
        logger.exception("Gemini project plan generation request failed")
        raise ProjectPlanGenerationError(
            "AI provider is temporarily unavailable. Please retry shortly."
        ) from e
    except json.JSONDecodeError as e:
        logger.exception("LLM response is not valid JSON")
        raise ProjectPlanGenerationError(
            "AI response could not be parsed as JSON. Please try again."
        ) from e
    except Exception as e:
        logger.exception("Unexpected error in project plan generation")
        raise ProjectPlanGenerationError(
            "Failed to generate project plan using AI."
        ) from e


# ── Prompt Construction ────────────────────────────────────────────────

def _build_prompt(request: ProjectPlanRequest) -> str:
    input_json = json.dumps({
        "project": request.project,
        "brief": request.brief,
        "architectureSubmission": request.architectureSubmission,
        "uiuxSubmission": request.uiuxSubmission,
        "planningTeam": request.planningTeam,
    }, indent=2)

    # Generate schema as JSON string (Pydantic v2)
    schema_dict = ProjectPlanResponse.model_json_schema()
    schema_json = json.dumps(schema_dict, indent=2)

    prompt = f"""
You are a scrum master agent responsible for creating a detailed implementation plan.

You are given the following project details:

{input_json}

Your task is to generate a complete project plan, including:
- A summary of the overall plan.
- A list of assumptions.
- A timeline with phases and estimated weeks.
- A list of milestones, each with acceptance criteria.
- A list of tasks, each linked to a milestone, with priority, required skills, estimated hours, and acceptance criteria.
- Dependencies between tasks (use dependencyType "blocks" by default).
- A team plan with recommended roles and suggested team size.
- A risk register with risks, impacts, and mitigations.
- A project specification containing architecture, design system, API contract, data model, and conventions.

Important rules:
- Use only these priority values: low, medium, high, urgent.
- Use only these task status values: todo, blocked, in_progress, review, changes_requested, done, cancelled (set default to "todo").
- Use only these dependency types: blocks, related, after (prefer "blocks").
- Every task must reference an existing milestone via `milestoneClientKey`.
- Every dependency must reference existing task `clientKey`s.
- All `clientKey` values must be unique across milestones and tasks.
- Do not create circular dependencies.
- The response must be valid JSON, with no markdown, no extra text.
- The response must match the exact structure described below.

Output JSON schema:
{schema_json}

Now generate the plan for this project.
"""
    return prompt


# ── Gemini Helper ──────────────────────────────────────────────────────

def _get_model_candidates() -> List[str]:
    primary = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
    models = [primary] + [m.strip() for m in fallbacks if m.strip()]
    return list(dict.fromkeys(models))


def _generate_plan_response(client, prompt_text: str):
    models = _get_model_candidates()
    if not models:
        raise ProjectPlanGenerationError("No Gemini model configured.")

    last_model = models[-1]
    for model in models:
        try:
            return client.models.generate_content(
                model=model,
                contents=[prompt_text],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",  # Only JSON, no schema
                    temperature=0.3,
                    top_k=1,
                    top_p=0.1,
                    http_options=types.HttpOptions(timeout=int(GENAI_TIMEOUT * 1000)),
                ),
            )
        except errors.APIError as exc:
            if model == last_model:
                raise
            logger.warning(
                "Gemini plan generation failed with model '%s'; trying fallback: %s",
                model,
                exc,
            )
    raise ProjectPlanGenerationError("All Gemini models failed.")