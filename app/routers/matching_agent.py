from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/agents", tags=["Matching Agent"])


class TaskForMatching(BaseModel):
    taskId: str
    title: str
    description: str
    requiredSkills: list[str] = Field(default_factory=list)
    preferredSkills: list[str] = Field(default_factory=list)
    estimatedHours: int
    allocatedBudget: float
    experienceLevel: str | None = None
    searchCriteria: str | None = None


class CandidateForMatching(BaseModel):
    freelancerId: str
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    hourlyRate: float | None = None
    avgRating: float | None = None
    ratingsCount: int = 0
    vectorScore: float | None = None
    metadata: dict[str, Any] | None = None


class MatchTaskRequest(BaseModel):
    task: TaskForMatching
    candidates: list[CandidateForMatching]


@router.post("/match-task")
def matching_agent(request: MatchTaskRequest):
    # Later: AI reranks candidates.
    # Sprint 1: return mock/best candidate.
    first_candidate = request.candidates[0] if request.candidates else None

    return {
        "selectedFreelancerId": first_candidate.freelancerId if first_candidate else None,
        "candidates": [
            {
                "freelancerId": candidate.freelancerId,
                "finalScore": (
                    candidate.vectorScore if candidate.vectorScore is not None else 0.75
                ),
                "rationale": "Candidate is a mock match for the task."
            }
            for candidate in request.candidates
        ]
    }
