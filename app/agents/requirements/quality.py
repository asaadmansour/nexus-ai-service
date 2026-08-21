import re
from typing import Any


USER_REQUIRED_BRIEF_FIELDS = [
    "mainGoal",
    "targetUsers",
    "coreFeatures",
    "platforms",
    "solutionType",
    "scopeDetails",
    "integrations",
    "adminNeeds",
    "deliverables",
]


UNCERTAIN_ANSWERS = {
    "idk",
    "i do not know",
    "i don't know",
    "i dont know",
    "not sure",
    "not sure yet",
    "notsure",
    "unknown",
    "whatever",
    "anything",
    "you choose",
    "you decide",
    "no idea",
    "not decided",
    "no preference",
    "no preferences",
    "tbd",
}


QUESTION_PREFIX = re.compile(
    r"^(?:what|which|why|how|who|when|where|can\s+you|could\s+you|"
    r"should\s+i|do\s+i|does\s+it|explain|tell\s+me\s+about|like\s+what)\b",
    re.IGNORECASE,
)


def get_brief_scope_gaps(fields: dict[str, Any] | None) -> list[str]:
    values = fields if isinstance(fields, dict) else {}
    return [
        field
        for field in USER_REQUIRED_BRIEF_FIELDS
        if not is_brief_scope_field_complete(field, values.get(field))
    ]


def is_brief_scope_field_complete(field: str, value: Any) -> bool:
    items = _normalized_items(value)
    blocker = _is_uncertain_text if field == "mainGoal" else _is_non_answer_text
    if not items or all(blocker(item) for item in items):
        return False

    if field == "mainGoal":
        return any(
            len(item) >= 8
            and "?" not in item
            and re.search(
                r"\b(?:sell|buy|book|manage|track|show|display|explain|describe|collect|inform|"
                r"market|promote|reduce|automate|help|allow|enable|connect|order|"
                r"reserve|schedule|learn|contact|generate|receive|share|find|"
                r"compare|request|provide|present)\b",
                item,
                re.IGNORECASE,
            )
            is not None
            for item in items
        )
    if field == "targetUsers":
        return any(
            re.fullmatch(
                r"(?:user|users|people|everyone|anyone|all|general public|not sure)",
                item,
                re.IGNORECASE,
            )
            is None
            for item in items
        )
    if field == "coreFeatures":
        return any(
            re.fullmatch(
                r"(?:app|website|mobile website|mobile app|platform|system|"
                r"basic features?|standard features?|everything|something simple)",
                item,
                re.IGNORECASE,
            )
            is None
            for item in items
        )
    if field == "platforms":
        return any(
            re.search(
                r"\b(?:website|web app|web|ios|android|mobile app|native app|"
                r"desktop|tablet|responsive)\b",
                item,
                re.IGNORECASE,
            )
            for item in items
        )
    if field == "solutionType":
        return any(
            re.search(
                r"\b(?:landing page|single[ -]page|marketing website|multi[ -]page|"
                r"responsive website|website|web app|mobile app|native app|ios|"
                r"android|desktop app|portal|dashboard)\b",
                item,
                re.IGNORECASE,
            )
            for item in items
        )
    if field == "scopeDetails":
        return (
            len(items) >= 2 and all(not _is_non_answer_text(item) for item in items)
        ) or any(_has_concrete_scope_detail(item) for item in items)
    if field == "integrations":
        return any(
            _is_explicit_none(item)
            or re.search(
                r"\b(?:payment|stripe|paypal|paymob|map|google|email|sms|whatsapp|"
                r"social login|analytics|api|webhook|crm|erp|existing system|"
                r"calendar|shipping|delivery|storage)\b",
                item,
                re.IGNORECASE,
            )
            is not None
            for item in items
        )
    if field == "adminNeeds":
        return any(
            _is_explicit_none(item)
            or re.search(
                r"\b(?:admin|dashboard|back office|manage|moderate|report|content|"
                r"orders?|users?|inventory|bookings?)\b",
                item,
                re.IGNORECASE,
            )
            is not None
            for item in items
        )
    if field == "deliverables":
        return any(
            re.search(
                r"\b(?:working|website|web app|mobile app|ios|android|source code|"
                r"repository|design|figma|prototype|deployment|live link|"
                r"documentation|handover|setup)\b",
                item,
                re.IGNORECASE,
            )
            is not None
            for item in items
        )
    return bool(items)


def is_requirements_guidance_request(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = _normalize(value)
    if not normalized:
        return False
    return (
        _is_non_answer_text(normalized)
        or "?" in normalized
        or QUESTION_PREFIX.search(normalized) is not None
        or re.search(
            r"\b(?:what do (?:you|u) suggest|recommend|suggest|help me "
            r"(?:choose|decide)|what do you mean)\b",
            normalized,
            re.IGNORECASE,
        )
        is not None
    )


def is_uncertain_answer(value: Any) -> bool:
    items = _normalized_items(value)
    return bool(items) and all(_is_uncertain_text(item) for item in items)


def _normalized_items(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for value_item in value for item in _normalized_items(value_item)]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [str(value)]
    if not isinstance(value, str):
        return []
    return [
        normalized
        for item in re.split(r",|;|\n|\band\b", value, flags=re.IGNORECASE)
        if (normalized := _normalize(item))
    ]


def _normalize(value: str) -> str:
    value = re.sub(r"[_-]+", " ", value.lower())
    value = re.sub(r"[.!]+$", "", value)
    return " ".join(value.split())


def _is_non_answer_text(value: str) -> bool:
    return (
        not value
        or _is_uncertain_text(value)
        or "?" in value
        or QUESTION_PREFIX.search(value) is not None
    )


def _is_uncertain_text(value: str) -> bool:
    return not value or value in UNCERTAIN_ANSWERS


def _is_explicit_none(value: str) -> bool:
    return (
        re.fullmatch(
            r"(?:none|no|not needed|n/?a|no integrations?|no admin(?: dashboard| area)?)",
            value,
            re.IGNORECASE,
        )
        is not None
    )


def _has_concrete_scope_detail(value: str) -> bool:
    if len(value) < 8 or _is_non_answer_text(value):
        return False
    if re.search(
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|single|"
        r"several|few)\s+(?:page|pages|screen|screens|section|sections|step|steps)\b",
        value,
        re.IGNORECASE,
    ):
        return True
    markers = re.findall(
        r"\b(?:home|about|contact|pricing|signup|sign up|login|browse|search|"
        r"catalog|product|cart|checkout|booking|profile|dashboard|order|track|"
        r"upload|form|content|gallery|faq|journey|workflow)\b",
        value,
        re.IGNORECASE,
    )
    return len(markers) >= 2
