import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
GENAI_TIMEOUT = 45.0


class ProjectQuoteEstimationError(RuntimeError):
    """Raised when quote estimation input or output is invalid."""


class ProjectQuoteRequest(BaseModel):
    project: Dict[str, Any]
    brief: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ProjectQuoteResponse(BaseModel):
    amount: float
    currency: str
    quoteStatus: str = "pending_customer"
    confidence: float = 0.7
    complexity: str
    rationale: str
    assumptions: List[str] = Field(default_factory=list)
    pricingSignals: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

    @field_validator("quoteStatus")
    @classmethod
    def validate_quote_status(cls, value: str) -> str:
        if value not in {"pending_customer", "out_of_budget"}:
            return "pending_customer"
        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        cleaned = value.strip().upper()
        return cleaned[:3] if cleaned else "EGP"

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


def estimate_project_quote(request: ProjectQuoteRequest) -> Dict[str, Any]:
    budget_min, budget_max = _budget_range(request.project)
    prompt = _build_prompt(request, budget_min, budget_max)

    try:
        client = genai.Client()
        response_text = _generate_quote_text(client, prompt)
        result = json.loads(_extract_json(response_text))
        validated = _normalize_quote_response(
            ProjectQuoteResponse(**result),
            request,
            budget_min,
            budget_max,
        )
        return validated.model_dump()
    except (ProjectQuoteEstimationError, ValidationError, json.JSONDecodeError):
        logger.exception("Project quote AI response was invalid; using fallback")
        return _fallback_quote(request, "AI quote response was invalid.")
    except errors.APIError:
        logger.exception("Gemini project quote request failed; using fallback")
        return _fallback_quote(request, "AI provider was unavailable.")
    except Exception:
        logger.exception("Unexpected project quote error; using fallback")
        return _fallback_quote(request, "Quote estimator used deterministic fallback.")


def _normalize_quote_response(
    quote: ProjectQuoteResponse,
    request: ProjectQuoteRequest,
    budget_min: float,
    budget_max: float,
) -> ProjectQuoteResponse:
    quote.amount = _round_money(_clamp(quote.amount, budget_min, budget_max))
    quote.currency = quote.currency or _currency(request.project)
    quote.quoteStatus = "pending_customer"
    if not quote.rationale.strip():
        quote.rationale = "Estimated from the confirmed requirements and budget range."
    if not quote.assumptions:
        quote.assumptions = _default_assumptions()
    if not quote.pricingSignals:
        quote.pricingSignals = _fallback_pricing_signals(request)
    return quote


def _build_prompt(request: ProjectQuoteRequest, budget_min: float, budget_max: float) -> str:
    input_json = json.dumps(
        {
            "project": request.project,
            "brief": request.brief or {},
            "budgetRange": {
                "min": budget_min,
                "max": budget_max,
                "currency": _currency(request.project),
            },
        },
        indent=2,
        default=str,
    )
    schema_json = json.dumps(ProjectQuoteResponse.model_json_schema(), indent=2)

    return f"""
You are the Nexus AI project quote estimator.

Goal:
Estimate the final escrow price for the whole project after requirements are
confirmed and before architect/UIUX matching starts. Use the project size,
scope, platforms, features, timeline pressure, team needs, and current market
signals from web search when available.

Input:
{input_json}

Rules:
- Return JSON only. No markdown. No extra text.
- The final `amount` MUST be inside the customer budget range:
  minimum {budget_min}, maximum {budget_max}.
- Use `quoteStatus: "pending_customer"` unless the budget range itself is
  unusable. Prefer staying inside the range and explain tight budget risk in
  assumptions/pricingSignals.
- The amount is a customer-facing final estimate, not an hourly rate.
- Include architecture and UI/UX planning effort because those are mandatory
  before implementation.
- Cite market signals briefly in `pricingSignals` and put source URLs/domains in
  `sources` when search grounding provides them.
- Be conservative: do not underprice complex multi-platform projects.
- Match this exact output schema:
{schema_json}
"""


def _generate_quote_text(client, prompt_text: str) -> str:
    models = _get_model_candidates()
    if not models:
        raise ProjectQuoteEstimationError("No Gemini model configured.")

    last_error: Exception | None = None
    for model in models:
        for grounded in (True, False):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[prompt_text],
                    config=_quote_generation_config(grounded),
                )
                if response.text:
                    return response.text
            except (TypeError, AttributeError) as exc:
                last_error = exc
                if grounded:
                    logger.warning(
                        "Google Search grounding not available in this SDK; retrying without it: %s",
                        exc,
                    )
                    continue
                raise
            except errors.APIError as exc:
                last_error = exc
                logger.warning(
                    "Gemini quote generation failed with model '%s' grounded=%s: %s",
                    model,
                    grounded,
                    exc,
                )
                continue

    if last_error:
        raise last_error
    raise ProjectQuoteEstimationError("All Gemini quote attempts returned empty output.")


def _quote_generation_config(grounded: bool):
    kwargs: Dict[str, Any] = {
        "response_mime_type": "application/json",
        "temperature": 0.2,
        "top_k": 1,
        "top_p": 0.1,
        "http_options": types.HttpOptions(timeout=int(GENAI_TIMEOUT * 1000)),
    }
    if grounded:
        kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    return types.GenerateContentConfig(**kwargs)


def _get_model_candidates() -> List[str]:
    primary = os.getenv("GEMINI_QUOTE_MODEL") or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
    models = [primary] + [model.strip() for model in fallbacks if model.strip()]
    return list(dict.fromkeys(models))


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
    return cleaned[start : end + 1]


def _fallback_quote(request: ProjectQuoteRequest, reason: str) -> Dict[str, Any]:
    budget_min, budget_max = _budget_range(request.project)
    brief = request.brief or {}
    feature_count = _count_items(brief.get("coreFeatures"))
    platform_count = max(1, _count_items(brief.get("platforms")))
    deliverable_count = _count_items(brief.get("deliverables"))
    team_size = _to_number(brief.get("suggestedTeamSize")) or 2.0
    complexity_score = min(
        1.0,
        0.2
        + min(feature_count, 8) * 0.06
        + min(platform_count, 3) * 0.08
        + min(deliverable_count, 5) * 0.04
        + min(team_size, 8) * 0.025
        + _deadline_pressure(request.project.get("deadline")),
    )
    factor = min(0.92, max(0.55, 0.52 + complexity_score * 0.35))
    amount = _round_money(budget_min + (budget_max - budget_min) * factor)
    complexity = "high" if complexity_score >= 0.72 else "medium" if complexity_score >= 0.45 else "low"

    quote = ProjectQuoteResponse(
        amount=amount,
        currency=_currency(request.project),
        quoteStatus="pending_customer",
        confidence=0.55,
        complexity=complexity,
        rationale="Estimated from the confirmed requirements and budget range.",
        assumptions=_default_assumptions() + [reason],
        pricingSignals=_fallback_pricing_signals(request),
        sources=["Nexus deterministic project quote fallback"],
    )
    return quote.model_dump()


def _fallback_pricing_signals(request: ProjectQuoteRequest) -> List[str]:
    brief = request.brief or {}
    return [
        f"{_count_items(brief.get('coreFeatures')) or 'Several'} core feature area(s) in scope.",
        f"{max(1, _count_items(brief.get('platforms')))} platform target(s) included.",
        "Mandatory architecture and UI/UX planning included before implementation.",
    ]


def _default_assumptions() -> List[str]:
    return [
        "The first release follows the confirmed brief without major scope expansion.",
        "The final escrow amount funds planning and implementation for the agreed scope.",
        "Any major change after payment should be handled as a revision or change request.",
    ]


def _budget_range(project: Dict[str, Any]) -> tuple[float, float]:
    minimum = max(0.0, _to_number(project.get("budgetMin")) or 0.0)
    maximum = _to_number(project.get("budgetMax"))
    if maximum is None:
        maximum = minimum
    maximum = max(minimum, maximum)
    return minimum, maximum


def _currency(project: Dict[str, Any]) -> str:
    raw = str(project.get("currency") or "EGP").strip().upper()
    return raw[:3] if raw else "EGP"


def _to_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _count_items(value: Any) -> int:
    if isinstance(value, list):
        return len([item for item in value if str(item).strip()])
    if isinstance(value, str) and value.strip():
        return len([item for item in re.split(r",|;|\n|\band\b", value) if item.strip()])
    return 0


def _deadline_pressure(value: Any) -> float:
    if not value:
        return 0.0
    try:
        raw = str(value).replace("Z", "+00:00")
        deadline = datetime.fromisoformat(raw)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        days = (deadline - datetime.now(timezone.utc)).total_seconds() / 86400
    except ValueError:
        return 0.0
    if days <= 14:
        return 0.12
    if days <= 30:
        return 0.07
    if days <= 60:
        return 0.03
    return 0.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return minimum
    return min(maximum, max(minimum, value))


def _round_money(value: float) -> float:
    return round(float(value), 2)
