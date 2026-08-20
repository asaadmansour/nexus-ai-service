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

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
GENAI_TIMEOUT = 45.0


class ProjectQuoteEstimationError(RuntimeError):
    """Raised when quote estimation input or output is invalid."""


class ProjectQuoteRequest(BaseModel):
    project: Dict[str, Any]
    brief: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RoleEstimate(BaseModel):
    roleKey: str
    people: int = Field(ge=1, le=12)
    hoursEach: float = Field(gt=0)
    hourlyRate: float = Field(gt=0)
    subtotal: float = Field(gt=0)


class ProjectQuoteResponse(BaseModel):
    amount: float
    recommendedMinimum: float
    budgetGap: float = 0
    roleEstimates: List[RoleEstimate] = Field(min_length=4)
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
    role_total = sum(
        estimate.people * estimate.hoursEach * estimate.hourlyRate
        for estimate in quote.roleEstimates
    )
    for estimate in quote.roleEstimates:
        estimate.subtotal = _round_money(
            estimate.people * estimate.hoursEach * estimate.hourlyRate
        )
    market_minimum = role_total / 0.9
    quote.recommendedMinimum = _round_money(
        max(quote.recommendedMinimum, quote.amount, market_minimum)
    )
    quote.amount = _round_money(max(budget_min, quote.recommendedMinimum))
    quote.budgetGap = _round_money(max(quote.recommendedMinimum - budget_max, 0))
    quote.currency = quote.currency or _currency(request.project)
    quote.quoteStatus = "out_of_budget" if quote.budgetGap > 0 else "pending_customer"
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
- Estimate realistic role hours, people, and hourly market rates first. Do not
  force the estimate inside the customer's budget.
- `recommendedMinimum` is the market-based total including a 10% Nexus fee.
- `amount` is at least recommendedMinimum. If recommendedMinimum exceeds the
  customer's maximum {budget_max}, set quoteStatus to "out_of_budget" and set
  budgetGap to the exact difference. Otherwise use "pending_customer".
- roleEstimates must cover principal_reviewer, architect, ui_ux, and implementation.
  The implementation row may contain multiple people. Its subtotal is
  people × hoursEach × hourlyRate.
- The amount is a customer-facing final estimate, not an hourly rate.
- Include architecture and UI/UX planning effort because those stages are mandatory,
  but scale their effort to the actual complexity. A trivial single-screen project gets
  a minimal planning package, not enterprise architecture and prototype pricing.
- Cite market signals briefly in `pricingSignals` and put source URLs/domains in
  `sources` when search grounding provides them.
- Be conservative: do not underprice complex multi-platform projects.
- Treat brief.requirementProfile and its cleaned feature list as authoritative for
  scope sizing. Ignore conversational questions, placeholders, uncertainty, and
  handover deliverables that may still appear in legacy summary or brief text.
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
    requirement_profile = brief.get("requirementProfile") or {}
    planning_complexity = (
        requirement_profile.get("complexity")
        if isinstance(requirement_profile, dict)
        else None
    )
    team_size = _to_number(brief.get("suggestedTeamSize")) or (
        1.0 if planning_complexity == "trivial" else 2.0
    )
    complexity_score = min(
        1.0,
        (0.05 if planning_complexity == "trivial" else 0.2)
        + min(feature_count, 8) * 0.06
        + min(platform_count, 3) * 0.08
        + min(deliverable_count, 5) * 0.04
        + min(team_size, 8) * 0.025
        + _deadline_pressure(request.project.get("deadline")),
    )
    complexity = "high" if complexity_score >= 0.72 else "medium" if complexity_score >= 0.45 else "low"
    complexity_key = (
        "trivial"
        if planning_complexity == "trivial"
        else "complex" if complexity == "high" else "standard"
    )
    hours = {
        "trivial": {"reviewer": 2, "architect": 2, "uiux": 2, "implementation": 8},
        "standard": {"reviewer": 12, "architect": 16, "uiux": 18, "implementation": 160},
        "complex": {"reviewer": 28, "architect": 36, "uiux": 40, "implementation": 480},
    }[complexity_key]
    workers = max(1, min(8, round(team_size)))
    role_estimates = [
        _role_estimate("principal_reviewer", 1, hours["reviewer"], _market_rate("MARKET_RATE_PRINCIPAL_REVIEWER", 650)),
        _role_estimate("architect", 1, hours["architect"], _market_rate("MARKET_RATE_ARCHITECT", 550)),
        _role_estimate("ui_ux", 1, hours["uiux"], _market_rate("MARKET_RATE_UI_UX", 450)),
        _role_estimate("implementation", workers, _ceil_div(hours["implementation"], workers), _market_rate("MARKET_RATE_DEVELOPER", 400)),
    ]
    labor_total = sum(item.subtotal for item in role_estimates)
    recommended_minimum = _round_money(labor_total / 0.9)
    amount = _round_money(max(budget_min, recommended_minimum))
    budget_gap = _round_money(max(recommended_minimum - budget_max, 0))

    quote = ProjectQuoteResponse(
        amount=amount,
        recommendedMinimum=recommended_minimum,
        budgetGap=budget_gap,
        roleEstimates=role_estimates,
        currency=_currency(request.project),
        quoteStatus="out_of_budget" if budget_gap > 0 else "pending_customer",
        confidence=0.55,
        complexity=complexity,
        rationale="Estimated from role hours and market-rate assumptions before comparison with the customer budget.",
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
        "Proportionate architecture and UI/UX planning included before implementation.",
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


def _market_rate(name: str, fallback: float) -> float:
    return _to_number(os.getenv(name)) or fallback


def _role_estimate(
    role_key: str,
    people: int,
    hours_each: float,
    hourly_rate: float,
) -> RoleEstimate:
    return RoleEstimate(
        roleKey=role_key,
        people=people,
        hoursEach=hours_each,
        hourlyRate=hourly_rate,
        subtotal=_round_money(people * hours_each * hourly_rate),
    )


def _ceil_div(value: float, divisor: int) -> int:
    return int((value + divisor - 1) // divisor)
