from typing import Any, TypedDict


class RequirementsState(TypedDict, total=False):
    # Input from Nest.
    projectId: str
    briefId: str
    latestMessage: str
    currentBrief: dict[str, Any]
    recentMessages: list[dict[str, Any]]

    # Values produced by the LangGraph nodes.
    knownFields: dict[str, Any]
    pendingField: str | None
    useFastPath: bool
    fastPathUsed: bool
    fastPathReason: str | None
    extractionSource: str
    extractedFields: dict[str, Any]
    assistantReply: str | None
    mergedBrief: dict[str, Any]
    missingFields: list[str]
    completionPercentage: int
    isComplete: bool
    nextQuestion: str | None
    nextQuestionField: str | None


REQUIRED_BRIEF_FIELDS = [
    "projectType",
    "businessDomain",
    "mainGoal",
    "targetUsers",
    "coreFeatures",
    "platforms",
    "solutionType",
    "scopeDetails",
    "integrations",
    "adminNeeds",
    "budget",
    "deadline",
    "deliverables",
    "constraintsPreferences",
    "clientBackground",
    "suggestedTeamSize",
    "experienceLevel",
    "experienceMinYears",
]
