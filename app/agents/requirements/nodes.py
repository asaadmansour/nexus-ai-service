from typing import Any

from app.agents.requirements.llm import extract_requirements_with_llm
from app.agents.requirements.intent import classify_requirements_message
from app.agents.requirements.quality import (
    USER_REQUIRED_BRIEF_FIELDS,
    get_brief_scope_gaps,
    is_brief_scope_field_complete,
    is_requirements_guidance_request,
    is_uncertain_answer,
)
from app.agents.requirements.state import REQUIRED_BRIEF_FIELDS, RequirementsState


QUESTION_BY_FIELD = {
    "businessDomain": "Nice, that gives me a better starting point. What kind of business or domain is this for? For example bakery, clinic, tutoring, ecommerce, logistics, or anything similar.",
    "mainGoal": "That helps. What is the main outcome you want from this project for your business? For example sell online, manage bookings, track stock, reduce manual work, or reach more customers.",
    "targetUsers": "Got it. Who will use it, and what should each group be able to do? For example customers place orders, staff manage stock, and admins track sales.",
    "coreFeatures": "Great. What are the must-have features for the first version? Short bullets are fine, like online payments, product catalog, order tracking, dashboard, notifications, or stock management.",
    "platforms": "Makes sense. Where should this run: website, mobile app, admin dashboard, or all of them? If you are not sure, tell me how people will use it and I will help translate that.",
    "solutionType": "To price this correctly, which best describes what you need: a single landing page, a multi-page marketing website, a web application, or a real iOS/Android mobile app? A mobile-friendly website still counts as a website.",
    "scopeDetails": "What should the first version contain? A rough page or screen count plus the main user journey is enough—for example, one landing page with five sections, or ten screens covering signup, browsing, checkout, and order tracking.",
    "integrations": "Does the first version connect to anything external, such as payments, maps, email/SMS, social login, analytics, or an existing system? You can simply say \"none\".",
    "adminNeeds": "Will your team need a private admin area to manage content, users, orders, or reports? If not, say \"no admin dashboard\".",
    "deliverables": "Good. What final things should be handed over when the work is done? For example a working website, mobile app, admin dashboard, source code, or deployment/setup help. If you are unsure, I can recommend a handover package for this project.",
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


def classify_message_node(state: RequirementsState) -> dict[str, Any]:
    current_brief = state.get("currentBrief", {})
    conversation_mode = (
        current_brief.get("conversationMode")
        if isinstance(current_brief, dict)
        else None
    )
    intent = classify_requirements_message(
        state.get("latestMessage", ""),
        conversation_mode=conversation_mode,
        pending_field=state.get("pendingField"),
    )
    return {"messageIntent": intent}


def respond_without_llm_node(state: RequirementsState) -> dict[str, Any]:
    intent = state.get("messageIntent")
    pending_field = state.get("pendingField")
    known_fields = state.get("knownFields", {})
    missing_fields = get_brief_scope_gaps(
        known_fields if isinstance(known_fields, dict) else {}
    )
    next_field = (
        pending_field
        if isinstance(pending_field, str) and pending_field in missing_fields
        else (missing_fields[0] if missing_fields else None)
    )
    next_question = QUESTION_BY_FIELD.get(next_field, "") if next_field else ""

    if intent == "security":
        prefix = (
            "I can help define your project, but I can’t change my role or reveal "
            "private instructions."
        )
        reply_mode = "security_boundary"
    elif intent == "out_of_scope":
        prefix = (
            "I can only help shape this project’s requirements, so I won’t answer "
            "unrelated questions here."
        )
        reply_mode = "scope_boundary"
    elif intent == "social":
        prefix = "You’re welcome—let’s keep shaping the first version."
        reply_mode = "social"
    elif intent == "initial_greeting":
        current_brief = state.get("currentBrief", {})
        project_context = (
            current_brief.get("projectContext", {})
            if isinstance(current_brief, dict)
            else {}
        )
        project_title = (
            project_context.get("title")
            if isinstance(project_context, dict)
            else None
        )
        title_text = (
            f' for “{str(project_title).strip()[:80]}”'
            if project_title and str(project_title).strip()
            else ""
        )
        prefix = (
            "Hi! I’ll help turn your idea"
            f"{title_text} into a clear, priceable first release."
        )
        reply_mode = "initial_greeting"
    else:
        guidance_field = _resolve_guidance_field(
            state.get("latestMessage", ""), next_field
        )
        field = guidance_field or next_field
        return {
            "useFastPath": True,
            "fastPathUsed": True,
            "fastPathReason": "deterministic_requirements_guidance",
            "extractionSource": "guidance",
            "extractedFields": {},
            "assistantReply": _build_guidance_reply(field) if field else None,
            "replyMode": "guidance",
        }

    assistant_reply = " ".join(part for part in (prefix, next_question) if part)
    return {
        "useFastPath": True,
        "fastPathUsed": True,
        "fastPathReason": f"deterministic_{intent}",
        "extractionSource": "scope_guard" if intent in {"security", "out_of_scope"} else "deterministic",
        "extractedFields": {},
        "assistantReply": assistant_reply,
        "replyMode": reply_mode,
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
    guidance_field = _resolve_guidance_field(latest_message, pending_field)
    if isinstance(pending_field, str) and is_requirements_guidance_request(latest_message):
        extracted_fields = dict(extracted_fields)
        extracted_fields.pop(pending_field, None)
    if guidance_field and (
        is_uncertain_answer(latest_message)
        or _is_definition_request(latest_message)
        or not assistant_reply
    ):
        assistant_reply = _build_guidance_reply(guidance_field)

    return {
        "useFastPath": False,
        "fastPathUsed": False,
        "fastPathReason": None,
        "extractionSource": "llm",
        "extractedFields": extracted_fields,
        "assistantReply": assistant_reply,
        "replyMode": "project_answer" if state.get("messageIntent") == "project_question" else "requirements_progress",
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

    missing_fields = get_brief_scope_gaps(merged_brief)
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
        return {
            "nextQuestion": None,
            "nextQuestionField": None,
            "pendingField": None,
            "assistantReply": (
                "Thanks—the first-release scope is complete. Review the brief, "
                "then confirm it to generate your quote."
            ),
            "replyMode": "complete",
        }

    next_field = missing_fields[0]
    next_question = QUESTION_BY_FIELD.get(
        next_field,
        "Can you share more details about the project?",
    )
    assistant_reply = state.get("assistantReply")
    reply_mode = state.get("replyMode")
    if reply_mode == "project_answer" and isinstance(assistant_reply, str):
        # The model answers the in-scope question; the graph owns progression.
        assistant_reply = f"{assistant_reply.rstrip()} {next_question}"
    elif not isinstance(assistant_reply, str) or not assistant_reply.strip():
        assistant_reply = next_question
    return {
        "nextQuestionField": next_field,
        "pendingField": next_field,
        "nextQuestion": next_question,
        "assistantReply": assistant_reply,
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
        if key in REQUIRED_BRIEF_FIELDS
        and _has_value(value)
        and (
            key not in USER_REQUIRED_BRIEF_FIELDS
            or is_brief_scope_field_complete(key, value)
        )
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


def _build_guidance_reply(field: str) -> str:
    replies = {
        "mainGoal": "No problem. The goal is the business result, not the technology. Common choices are getting leads, selling online, reducing manual work, or helping customers self-serve. Which result matters most for this first version?",
        "targetUsers": "No problem. Think about the people who will actually use it: customers, staff, admins, or a specific group. Who completes the main action in the first version?",
        "coreFeatures": "No problem. Features are the actions the product must support. For a small website that might be reading key information and sending an enquiry; for an app it could include accounts, booking, or checkout. What is the single most important action a user must complete?",
        "platforms": "No problem. A responsive website opens in a browser and works on phones; a mobile app is installed from iOS or Android stores and costs more to build. I usually recommend starting with a responsive website unless app-only features are essential. Which should we use?",
        "solutionType": "No problem. A landing page is one focused page, a marketing website has several information pages, a web app supports accounts or workflows, and a mobile app is installed on iOS or Android. Which smallest option meets your first-release goal?",
        "scopeDetails": "No problem. A rough answer is enough: either give a page or screen count, or describe the path from opening the product to completing the main goal. For example, 'one page with five sections' or 'signup, browse, checkout, confirmation.' What is closest?",
        "integrations": "No problem. Integrations are outside services such as payments, maps, email or SMS, social login, analytics, or an existing business system. I recommend 'none' for the first version unless one is essential. Which do you need?",
        "adminNeeds": "No problem. An admin area is a private screen your team uses to manage content, users, orders, bookings, or reports. If nobody needs to manage changing data, I recommend no admin dashboard. Should we include one, and what would it manage?",
        "deliverables": "No problem. Deliverables are what you receive at handover. I recommend the working product, source code, deployment or a live link, and a short setup guide. Should I use that package?",
    }
    return replies.get(
        field,
        "No problem. I can explain the options and recommend the simplest one that meets your goal. What part would you like me to clarify?",
    )


def _resolve_guidance_field(value: Any, pending_field: Any) -> str | None:
    if not isinstance(value, str) or not is_requirements_guidance_request(value):
        return None
    normalized = " ".join(value.lower().replace("_", " ").split())
    markers = {
        "mainGoal": ("goal", "outcome", "business result"),
        "targetUsers": ("target user", "audience", "who will use"),
        "coreFeatures": ("feature", "functionality", "must have"),
        "platforms": ("platform", "website or app", "web or mobile"),
        "solutionType": ("solution type", "landing page", "web app"),
        "scopeDetails": ("scope", "page count", "screen count", "user journey"),
        "integrations": ("integration", "third party", "external service"),
        "adminNeeds": ("admin", "dashboard", "back office"),
        "deliverables": ("deliverable", "handover", "receive at the end"),
    }
    for field, field_markers in markers.items():
        if any(marker in normalized for marker in field_markers):
            return field
    if is_uncertain_answer(value) or any(
        marker in normalized
        for marker in (
            "suggest",
            "recommend",
            "help me choose",
            "help me decide",
            "what do you mean",
            "explain this",
        )
    ):
        return pending_field if isinstance(pending_field, str) else None
    return None


def _is_definition_request(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = " ".join(value.lower().split())
    return (
        "?" in normalized
        or normalized.startswith(
            (
                "what ",
                "which ",
                "why ",
                "how ",
                "can you explain",
                "could you explain",
                "explain ",
            )
        )
    )
