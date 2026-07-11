from typing import Any

from app.agents.requirements.llm import extract_requirements_with_llm
from app.agents.requirements.state import REQUIRED_BRIEF_FIELDS, RequirementsState

PROJECT_DERIVED_FIELDS = {"projectType", "budget", "deadline"}
USER_REQUIRED_BRIEF_FIELDS = [
    field for field in REQUIRED_BRIEF_FIELDS if field not in PROJECT_DERIVED_FIELDS
]


QUESTION_BY_FIELD = {
    "businessDomain": "What kind of business or domain is this for?",
    "mainGoal": "What is the main thing you want this project to achieve?",
    "targetUsers": "Who will use this most: customers, staff, admins, or another group?",
    "coreFeatures": "What are the must-have features you want in the first version?",
    "platforms": "Where should this run: website, mobile app, both, or something else?",
    "deliverables": "What final deliverables would feel complete to you, like a working website, mobile app, dashboard, setup help, or simply \"not sure\"?",
    "constraintsPreferences": "Any preferences or constraints we should respect, like colors, style, integrations, or things you want to avoid?",
    "clientBackground": "What is your background here: business owner, operations, non-technical founder, technical founder, or something else?",
    "suggestedTeamSize": "Do you already have a team size in mind, or should we suggest what fits the project?",
    "experienceLevel": "Do you prefer a junior, mid, senior, or expert freelancer, or should we decide based on the scope?",
    "experienceMinYears": "Do you have a minimum years-of-experience preference, or is there no preference?",
}


def prepare_brief_context_node(state: RequirementsState) -> dict[str, Any]:
    current_brief = state.get("currentBrief", {})
    if not isinstance(current_brief, dict):
        current_brief = {}

    known_fields = _extract_known_fields(current_brief)
    pending_field = _extract_pending_field(current_brief)

    if pending_field and pending_field not in REQUIRED_BRIEF_FIELDS:
        pending_field = None

    return {
        "knownFields": known_fields,
        "pendingField": pending_field,
    }


def extract_requirements_node(state: RequirementsState) -> dict[str, Any]:
    llm_result = extract_requirements_with_llm(state)
    extracted_fields = llm_result.get("extractedFields", {})
    assistant_reply = llm_result.get("assistantReply")

    if not isinstance(extracted_fields, dict):
        extracted_fields = {}
    if not isinstance(assistant_reply, str):
        assistant_reply = None

    return {
        "useFastPath": False,
        "fastPathUsed": False,
        "fastPathReason": None,
        "extractionSource": "llm",
        "extractedFields": extracted_fields,
        "assistantReply": assistant_reply,
    }


def merge_brief_node(state: RequirementsState) -> dict[str, Any]:
    known_fields = state.get("knownFields", {})
    extracted_fields = state.get("extractedFields", {})

    if not isinstance(known_fields, dict):
        known_fields = {}

    if not isinstance(extracted_fields, dict):
        extracted_fields = {}

    merged_brief = {
        **_filter_required_fields(known_fields),
        **_filter_required_fields(extracted_fields),
    }

    return {
        "mergedBrief": merged_brief,
        "extractedFields": merged_brief,
    }


def check_missing_fields_node(state: RequirementsState) -> dict[str, Any]:
    merged_brief = state.get("mergedBrief", {})
    if not isinstance(merged_brief, dict):
        merged_brief = {}

    missing_fields = [
        field
        for field in USER_REQUIRED_BRIEF_FIELDS
        if not _has_value(merged_brief.get(field))
    ]
    completed_fields = len(USER_REQUIRED_BRIEF_FIELDS) - len(missing_fields)
    completion_percentage = round(
        (completed_fields / len(USER_REQUIRED_BRIEF_FIELDS)) * 100
    )

    return {
        "missingFields": missing_fields,
        "completionPercentage": completion_percentage,
        "isComplete": not missing_fields,
    }


def choose_next_question_node(state: RequirementsState) -> dict[str, Any]:
    missing_fields = state.get("missingFields", [])

    if not missing_fields:
        return {"nextQuestion": None, "nextQuestionField": None, "pendingField": None}

    next_field = missing_fields[0]
    return {
        "nextQuestionField": next_field,
        "pendingField": next_field,
        "nextQuestion": QUESTION_BY_FIELD.get(
            next_field,
            "Can you share more details about the project?",
        ),
    }


def _extract_known_fields(current_brief: dict[str, Any]) -> dict[str, Any]:
    known_fields: dict[str, Any] = {}
    ai_decided = current_brief.get("aiDecided")
    if not isinstance(ai_decided, dict):
        ai_decided = {}

    sources = [
        current_brief,
        ai_decided.get("extractedFields"),
        current_brief.get("knownFields"),
    ]

    for source in sources:
        if not isinstance(source, dict):
            continue

        for field in REQUIRED_BRIEF_FIELDS:
            value = source.get(field)
            if _has_value(value):
                known_fields[field] = value

    return known_fields


def _extract_pending_field(current_brief: dict[str, Any]) -> str | None:
    ai_decided = current_brief.get("aiDecided")
    if not isinstance(ai_decided, dict):
        ai_decided = {}

    pending_field = (
        current_brief.get("pendingField")
        or current_brief.get("nextQuestionField")
        or ai_decided.get("pendingField")
        or ai_decided.get("nextQuestionField")
    )

    return pending_field if isinstance(pending_field, str) else None


def _filter_required_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in fields.items()
        if key in REQUIRED_BRIEF_FIELDS and _has_value(value)
    }


def _has_value(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, list):
        return any(_has_value(item) for item in value)

    if isinstance(value, dict):
        return bool(value)

    return True
