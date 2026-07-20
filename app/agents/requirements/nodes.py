from typing import Any

from app.agents.requirements.llm import extract_requirements_with_llm
from app.agents.requirements.state import REQUIRED_BRIEF_FIELDS, RequirementsState

PROJECT_DERIVED_FIELDS = {"projectType", "budget", "deadline"}
USER_REQUIRED_BRIEF_FIELDS = [
    field for field in REQUIRED_BRIEF_FIELDS if field not in PROJECT_DERIVED_FIELDS
]


QUESTION_BY_FIELD = {
    "businessDomain": "Nice, that gives me a better starting point. What kind of business or domain is this for? For example bakery, clinic, tutoring, ecommerce, logistics, or anything similar.",
    "mainGoal": "That helps. What is the main outcome you want from this project for your business? For example sell online, manage bookings, track stock, reduce manual work, or reach more customers.",
    "targetUsers": "Got it. Who will use it, and what should each group be able to do? For example customers place orders, staff manage stock, and admins track sales.",
    "coreFeatures": "Great. What are the must-have features for the first version? Short bullets are fine, like online payments, product catalog, order tracking, dashboard, notifications, or stock management.",
    "platforms": "Makes sense. Where should this run: website, mobile app, admin dashboard, or all of them? If you are not sure, tell me how people will use it and I will help translate that.",
    "deliverables": "Good. What final things should be handed over when the work is done? For example a working website, mobile app, admin dashboard, source code, deployment/setup help, or simply \"not sure\".",
    "constraintsPreferences": "Any preferences or constraints we should respect? This can be simple: colors, style, payment provider, delivery rules, integrations, language, or things you want to avoid.",
    "clientBackground": "To guide this properly, what is your background here? For example business owner, operations, non-technical founder, technical founder, or something else.",
    "suggestedTeamSize": "Do you already have a team size in mind, or should we suggest what fits the scope? It is completely okay to say \"not sure\".",
    "experienceLevel": "Do you prefer junior, mid, senior, or expert freelancers, or should we decide based on the project complexity?",
    "experienceMinYears": "Do you have a minimum years-of-experience preference, or should we keep it open and match based on skill scores instead?",
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

    pending_field = state.get("pendingField")
    latest_message = state.get("latestMessage", "")
    if (
        isinstance(pending_field, str)
        and _is_advice_request(latest_message)
        and _is_uncertain_value(extracted_fields.get(pending_field))
    ):
        extracted_fields = dict(extracted_fields)
        extracted_fields.pop(pending_field, None)

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
        "extractedFields": extracted_fields,
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


def _is_advice_request(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    normalized = " ".join(value.lower().split())
    advice_markers = (
        "what do you suggest",
        "what do u suggest",
        "what should",
        "what would you",
        "what would u",
        "recommend",
        "suggest",
        "help me choose",
        "what do you mean",
        "explain",
    )
    uncertainty_markers = ("idk", "i don't know", "i dont know", "not sure")
    return any(marker in normalized for marker in advice_markers) and (
        "?" in normalized
        or any(marker in normalized for marker in uncertainty_markers)
    )


def _is_uncertain_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    normalized = value.lower().replace("_", " ").replace("-", " ").strip()
    normalized = " ".join(normalized.split())
    return normalized in {
        "idk",
        "i do not know",
        "i don't know",
        "not sure",
        "not sure yet",
        "notsure",
        "no preference",
        "no preferences",
        "not decided",
    }
