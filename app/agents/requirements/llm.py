import json
import logging
import os
from typing import Any

from dotenv import load_dotenv

from app.agents.requirements.prompts import (
    REQUIREMENTS_SYSTEM_PROMPT,
    build_requirements_extraction_prompt,
)
from app.agents.requirements.state import REQUIRED_BRIEF_FIELDS, RequirementsState


load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
logger = logging.getLogger(__name__)

REQUIREMENT_FIELD_VALUE_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "array", "items": {"type": "string"}},
    ]
}

REQUIREMENTS_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "extractedFields": {
            "type": "object",
            "properties": {
                field: REQUIREMENT_FIELD_VALUE_SCHEMA
                for field in REQUIRED_BRIEF_FIELDS
            },
        },
        "assistantReply": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
    },
    "required": ["extractedFields", "assistantReply"],
}

FIELD_LABELS = {field.lower() for field in REQUIRED_BRIEF_FIELDS}
FIELD_LABELS.update(
    {
        "project_type",
        "business_domain",
        "main_goal",
        "target_users",
        "core_features",
        "constraints_preferences",
        "client_background",
        "suggested_team_size",
        "experience_level",
        "experience_min_years",
    }
)

_client: Any | None = None


def extract_requirements_with_llm(state: RequirementsState) -> dict[str, Any]:
    prompt = build_requirements_extraction_prompt(state)
    raw_text = _generate_json_text(prompt)
    parsed = _parse_json_object(raw_text)
    return _normalize_llm_result(parsed)


def _generate_json_text(prompt: str) -> str:
    client = _get_client()
    generation_config = _build_generation_config()
    generation_config[_get_response_schema_key(client)] = REQUIREMENTS_EXTRACTION_SCHEMA
    last_error: Exception | None = None

    for model in _get_model_candidates():
        try:
            return _generate_json_text_with_model(
                client,
                model,
                prompt,
                generation_config,
            )
        except _retryable_generation_errors() as exc:
            last_error = exc

    if last_error:
        raise last_error

    raise RuntimeError("No Gemini model is configured.")


def _generate_json_text_with_model(
    client: Any,
    model: str,
    prompt: str,
    generation_config: dict[str, Any],
) -> str:
    if hasattr(client, "models"):
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "system_instruction": REQUIREMENTS_SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                **generation_config,
            },
        )
        return getattr(response, "text", "") or ""

    interaction = client.interactions.create(
        model=model,
        system_instruction=REQUIREMENTS_SYSTEM_PROMPT,
        input=prompt,
        generation_config=generation_config,
        response_format={
            "type": "text",
            "mime_type": "application/json",
        },
    )
    return getattr(interaction, "output_text", "") or ""


def _get_response_schema_key(client: Any) -> str:
    return "response_schema" if hasattr(client, "models") else "responseSchema"


def _retryable_generation_errors() -> tuple[type[Exception], ...]:
    retryable_errors: list[type[Exception]] = [TimeoutError, ConnectionError]

    try:
        from google.genai.errors import ServerError
    except ImportError:
        pass
    else:
        retryable_errors.append(ServerError)

    try:
        import httpx
    except ImportError:
        pass
    else:
        retryable_errors.extend([httpx.TimeoutException, httpx.TransportError])

    return tuple(dict.fromkeys(retryable_errors))


def _get_model_candidates() -> list[str]:
    primary_model = os.getenv(
        "GEMINI_REQUIREMENTS_MODEL",
        os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    )
    fallback_models = os.getenv("GEMINI_REQUIREMENTS_FALLBACK_MODELS", "")

    models = [
        primary_model,
        *[
            model.strip()
            for model in fallback_models.split(",")
            if model.strip()
        ],
    ]

    return list(dict.fromkeys(models))


def _build_generation_config() -> dict[str, Any]:
    config: dict[str, Any] = {
        "temperature": 0,
        "max_output_tokens": int(
            os.getenv("GEMINI_REQUIREMENTS_MAX_OUTPUT_TOKENS", "1024")
        ),
    }

    thinking_level = os.getenv("GEMINI_THINKING_LEVEL")
    if thinking_level is not None and thinking_level.strip():
        config["thinking_level"] = thinking_level.strip()

    thinking_budget = os.getenv(
        "GEMINI_REQUIREMENTS_THINKING_BUDGET",
        os.getenv("GEMINI_THINKING_BUDGET", "0"),
    )
    if thinking_budget is not None and thinking_budget.strip():
        config["thinking_config"] = {
            "thinking_budget": int(thinking_budget),
            "include_thoughts": False,
        }

    return config


def _get_client() -> Any:
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is not installed in this Python environment."
        ) from exc

    _client = genai.Client(api_key=api_key)
    return _client


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse requirements model JSON response.", exc_info=True)
        return {}

    if not isinstance(parsed, dict):
        return {}

    return parsed


def _filter_allowed_fields(fields: dict[str, Any]) -> dict[str, Any]:
    allowed = set(REQUIRED_BRIEF_FIELDS)
    filtered: dict[str, Any] = {}

    for key, value in fields.items():
        if key not in allowed or _is_empty(value):
            continue

        if isinstance(value, str):
            clean_value = _clean_field_value(key, value)
            if clean_value:
                filtered[key] = clean_value
            continue

        if isinstance(value, int | float):
            filtered[key] = value
            continue

        if isinstance(value, list):
            clean_values = _clean_field_list(key, value)
            if clean_values:
                filtered[key] = clean_values

    return filtered


def _normalize_llm_result(parsed: dict[str, Any]) -> dict[str, Any]:
    extracted_fields = parsed.get("extractedFields")
    if not isinstance(extracted_fields, dict):
        extracted_fields = parsed

    return {
        "extractedFields": _filter_allowed_fields(extracted_fields),
        "assistantReply": _clean_assistant_reply(parsed.get("assistantReply")),
    }


def _clean_assistant_reply(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = " ".join(value.strip().split())
    return cleaned[:700] if cleaned else None


def _clean_field_list(field: str, values: list[Any]) -> list[str]:
    clean_values: list[str] = []

    for item in values:
        if not isinstance(item, str):
            continue

        if _is_field_label(item):
            break

        clean_value = _clean_field_value(field, item)
        if clean_value:
            clean_values.append(clean_value)

    return clean_values


def _clean_field_value(field: str, value: str) -> str | None:
    cleaned = value.strip()
    if not cleaned or _is_field_label(cleaned):
        return None

    lowered = cleaned.lower()
    earliest_marker_index: int | None = None
    for label in FIELD_LABELS:
        marker = f"{label}:"
        index = lowered.find(marker)
        if index == 0:
            return None
        if index > 0:
            if earliest_marker_index is None or index < earliest_marker_index:
                earliest_marker_index = index

    if earliest_marker_index is not None:
        cleaned = cleaned[:earliest_marker_index].strip(" ,;.-")

    if field == "targetUsers" and _looks_like_non_target_user_value(cleaned):
        return None

    return cleaned or None


def _is_field_label(value: str) -> bool:
    normalized = value.strip().lower().replace(" ", "").replace("_", "")
    normalized = normalized.rstrip(":")
    labels = {label.replace("_", "") for label in FIELD_LABELS}
    return normalized in labels


def _looks_like_non_target_user_value(value: str) -> bool:
    lowered = value.lower()
    blocked_fragments = (
        "business domain",
        "clinic management",
        "main goal",
        "booking appointments",
        "manage doctors",
        "manage branches",
        "payments",
        "schedules",
        "system should",
    )
    return any(fragment in lowered for fragment in blocked_fragments)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return not value
    if isinstance(value, int | float):
        return False
    return True
