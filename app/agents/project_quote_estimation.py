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

from app.agents.requirements.quality import get_brief_scope_gaps

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
    scope_gaps = get_brief_scope_gaps(request.brief or {})
    if scope_gaps:
        raise ValueError(
            "A reliable quote needs more confirmed scope: " + ", ".join(scope_gaps)
        )

    if _is_minimal_website_scope(request.brief or {}):
        return _fallback_quote(
            request,
            "A deterministic minimal-scope quote was used to prevent speculative work.",
        )

    _, budget_max = _budget_range(request.project)
    prompt = _build_prompt(request)

    try:
        client = genai.Client()
        response_text = _generate_quote_text(client, prompt)
        result = json.loads(_extract_json(response_text))
        validated = _normalize_quote_response(
            ProjectQuoteResponse(**result),
            request,
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
    budget_max: float,
) -> ProjectQuoteResponse:
    requested_currency = _currency(request.project)
    if quote.currency != requested_currency:
        raise ProjectQuoteEstimationError(
            f"Quote currency {quote.currency} does not match requested currency "
            f"{requested_currency}."
        )
    # The LLM contributes market evidence and explanation only. Customer money
    # comes from the deterministic fixed-package calculator, which prevents
    # salary/agency hourly-rate anchoring and makes the same confirmed scope
    # stable across model calls.
    fixed = ProjectQuoteResponse(**_fallback_quote(request, "Fixed marketplace package pricing."))
    fixed.confidence = quote.confidence
    fixed.rationale = quote.rationale.strip() or fixed.rationale
    fixed.assumptions = quote.assumptions or fixed.assumptions
    fixed.pricingSignals = quote.pricingSignals or fixed.pricingSignals
    fixed.sources = quote.sources or fixed.sources
    return fixed


_SCOPE_PERSON_HOUR_ENVELOPES = {
    "trivial": (6.0, 28.0),
    "small": (24.0, 110.0),
    "standard": (70.0, 320.0),
    "complex": (180.0, 1100.0),
}
_TIER_HOURS = {
    "trivial": {"reviewer": 1, "architect": 1, "uiux": 2, "implementation": 8},
    "small": {"reviewer": 2, "architect": 3, "uiux": 6, "implementation": 32},
    "standard": {"reviewer": 6, "architect": 10, "uiux": 12, "implementation": 96},
    "complex": {"reviewer": 14, "architect": 24, "uiux": 28, "implementation": 260},
}
_TIER_BASE_USD = {"trivial": 85.0, "small": 240.0, "standard": 650.0, "complex": 1600.0}
_TIER_ROLE_SHARES = {
    "trivial": {"reviewer": 0.12, "architect": 0.10, "uiux": 0.13, "implementation": 0.55},
    "small": {"reviewer": 0.10, "architect": 0.12, "uiux": 0.18, "implementation": 0.50},
    "standard": {"reviewer": 0.10, "architect": 0.15, "uiux": 0.15, "implementation": 0.50},
    "complex": {"reviewer": 0.12, "architect": 0.18, "uiux": 0.16, "implementation": 0.44},
}
def _build_prompt(request: ProjectQuoteRequest) -> str:
    scope_tier = _scope_tier(request.brief or {})
    pricing_project = {
        key: value
        for key, value in request.project.items()
        if key not in {"budgetMin", "budgetMax", "budget", "quotedAmount"}
    }
    input_json = json.dumps(
        {
            "project": pricing_project,
            "brief": request.brief or {},
            "currency": _currency(request.project),
            "scopeTier": scope_tier,
            "personHourEnvelope": _SCOPE_PERSON_HOUR_ENVELOPES[scope_tier],
        },
        indent=2,
        default=str,
    )
    schema_json = json.dumps(ProjectQuoteResponse.model_json_schema(), indent=2)

    quote_currency = _currency(request.project)

    return f"""
You are the Nexus AI project quote estimator.

CURRENCY — READ FIRST:
Every monetary value you return MUST be expressed in {quote_currency}. That
includes each roleEstimates[].hourlyRate, each subtotal, amount, and
recommendedMinimum. Do NOT convert between currencies. Do NOT quote rates in one
currency in your prose and a different currency in the numeric fields. If you
reason about market rates sourced in another currency, convert them once, state
the converted figure, and use {quote_currency} everywhere thereafter.

Goal:
Estimate the final escrow price for the whole project after requirements are
confirmed and before architect/UIUX matching starts. Use the project size,
scope, platforms, features, timeline pressure, team needs, and current market
signals from web search when available.

Input:
{input_json}

Rules:
- Return JSON only. No markdown. No extra text.
- Estimate a realistic fixed-scope freelance package first. Do not derive the
  customer total by multiplying rank, hours, or employment hourly rates, and do
  not force the estimate inside the customer's budget.
- `recommendedMinimum` is the market-based total including a 10% Nexus fee.
- `amount` equals recommendedMinimum. The customer's budget is intentionally absent
  from the pricing evidence so it cannot anchor the estimate. Return
  quoteStatus "pending_customer" and budgetGap 0; the platform compares the
  independently calculated amount with the customer's budget afterward.
- roleEstimates must cover principal_reviewer, architect, ui_ux, and implementation.
  The implementation row may contain multiple people. Role rows divide the
  fixed package internally; hourlyRate is an effective compatibility value,
  not the customer pricing formula.
- The amount is a customer-facing final estimate, not an hourly rate.
- Every rate and total is in {quote_currency}. State {quote_currency} explicitly
  in `rationale` and `assumptions` so the figures cannot be misread.
- Include architecture and UI/UX planning effort because those stages are mandatory,
  but scale their effort to the actual complexity. A trivial single-screen project gets
  a minimal planning package, not enterprise architecture and prototype pricing.
- Price as fixed-scope freelance work, not as salaried employment or agency retainers.
  Prioritize comparable fixed-price listings and project catalogs from Khamsat,
  Mostaql, Fiverr, and Upwork. Glassdoor, Wuzzuf, annual salaries, and generic agency
  price articles are not valid evidence for the customer-facing project total.
- Use these broad marketplace sanity anchors before currency conversion: a simple
  landing/static page is commonly about 50-200 USD; a small marketing site is
  commonly about 150-700 USD; a scoped custom web app is commonly about 500-2,500
  USD; complex multi-role platforms may exceed
  that. Adjust for the confirmed work, not the client's budget.
- Stay inside the supplied scopeTier personHourEnvelope. Do not add contingency work,
  enterprise documentation, extra platforms, or speculative features to consume it.
- Cite market signals briefly in `pricingSignals` and put source URLs/domains in
  `sources` when search grounding provides them.
- Be conservative: do not underprice complex multi-platform projects.
- Treat brief.requirementProfile and its cleaned feature list as authoritative for
  scope sizing. Ignore conversational questions, placeholders, uncertainty, and
  handover deliverables that may still appear in legacy summary or brief text.
- Treat "mobile-friendly website", "mobile website", and "responsive website" as
  one web platform, never as a separate native mobile application. Count a mobile
  application only when the confirmed scope explicitly mentions iOS, Android,
  native/cross-platform app development, Flutter, React Native, or app-store delivery.
- Use brief.solutionType, brief.scopeDetails, brief.integrations, and
  brief.adminNeeds as the primary sizing evidence. A single landing page with a few
  sections, no integrations, and no admin area must receive minimal hours. Do not
  add speculative ecommerce, authentication, dashboard, or mobile-app work.
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
    _, budget_max = _budget_range(request.project)
    brief = request.brief or {}
    feature_count = _count_items(brief.get("coreFeatures"))
    platform_count = max(1, _count_items(brief.get("platforms")))
    deliverable_count = _count_items(brief.get("deliverables"))
    scope_tier = _scope_tier(brief)
    if scope_tier == "trivial":
        platform_count = 1
    suggested_team_size = _to_number(brief.get("suggestedTeamSize"))
    default_workers = {"trivial": 1, "small": 1, "standard": 2, "complex": 3}[
        scope_tier
    ]
    max_workers = {"trivial": 1, "small": 1, "standard": 3, "complex": 5}[
        scope_tier
    ]
    team_size = min(max_workers, max(1, round(suggested_team_size or default_workers)))
    complexity_score = min(
        1.0,
        {"trivial": 0.05, "small": 0.16, "standard": 0.32, "complex": 0.62}[
            scope_tier
        ]
        + min(feature_count, 8) * 0.06
        + min(platform_count, 3) * 0.08
        + min(deliverable_count, 5) * 0.04
        + min(team_size, 8) * 0.025
        + _deadline_pressure(request.project.get("deadline")),
    )
    complexity = "high" if complexity_score >= 0.72 else "medium" if complexity_score >= 0.45 else "low"
    hours = _TIER_HOURS[scope_tier]
    workers = max(1, min(8, round(team_size)))
    currency = _currency(request.project)
    package_total = _fixed_package_total(request, scope_tier)
    shares = _TIER_ROLE_SHARES[scope_tier]
    role_estimates = [
        _fixed_role_estimate("principal_reviewer", 1, hours["reviewer"], package_total * shares["reviewer"]),
        _fixed_role_estimate("architect", 1, hours["architect"], package_total * shares["architect"]),
        _fixed_role_estimate("ui_ux", 1, hours["uiux"], package_total * shares["uiux"]),
        _fixed_role_estimate("implementation", workers, _ceil_div(hours["implementation"], workers), package_total * shares["implementation"]),
    ]
    recommended_minimum = package_total
    amount = recommended_minimum
    budget_gap = _round_money(max(recommended_minimum - budget_max, 0))

    quote = ProjectQuoteResponse(
        amount=amount,
        recommendedMinimum=recommended_minimum,
        budgetGap=budget_gap,
        roleEstimates=role_estimates,
        currency=currency,
        quoteStatus="out_of_budget" if budget_gap > 0 else "pending_customer",
        confidence=0.55,
        complexity=complexity,
        rationale="Fixed-scope freelance package estimated from the confirmed product scope; effort assumptions only divide the package among delivery roles.",
        assumptions=_default_assumptions() + [reason],
        pricingSignals=_fallback_pricing_signals(request),
        sources=[
            "https://khamsat.com/programming/custom-website-development",
            "https://mostaql.com/projects/development",
            "https://www.fiverr.com/categories/programming-tech",
            "https://www.upwork.com/services/",
        ],
    )
    return quote.model_dump()


def _is_minimal_website_scope(brief: Dict[str, Any]) -> bool:
    solution_type = _normalized_scope_text(brief.get("solutionType"))
    platforms = _normalized_scope_text(brief.get("platforms"))
    integrations = _normalized_scope_text(brief.get("integrations"))
    admin_needs = _normalized_scope_text(brief.get("adminNeeds"))

    minimal_solution = any(
        marker in solution_type
        for marker in ("landing page", "single page", "single-page", "static website")
    )
    explicit_native_app = any(
        marker in f"{solution_type} {platforms}"
        for marker in (
            "mobile app",
            "native app",
            "ios app",
            "android app",
            "flutter",
            "react native",
        )
    )
    no_integrations = integrations in {"none", "no", "not needed", "n/a"} or bool(
        re.search(r"\bno integrations?\b", integrations)
    )
    no_admin = admin_needs in {"none", "no", "not needed", "n/a"} or bool(
        re.search(r"\bno admin(?: dashboard| area)?\b", admin_needs)
    )
    return minimal_solution and not explicit_native_app and no_integrations and no_admin


def _scope_tier(brief: Dict[str, Any]) -> str:
    if _is_minimal_website_scope(brief):
        return "trivial"

    solution = _normalized_scope_text(brief.get("solutionType"))
    platforms = _normalized_scope_text(brief.get("platforms"))
    scope = _normalized_scope_text(brief.get("scopeDetails"))
    integrations = _normalized_scope_text(brief.get("integrations"))
    admin = _normalized_scope_text(brief.get("adminNeeds"))
    features = _count_items(brief.get("coreFeatures"))
    platform_count = max(1, _count_items(brief.get("platforms")))
    page_count = _scope_count(scope)
    native_app = bool(
        re.search(
            r"\b(?:mobile app|native app|ios|android|flutter|react native)\b",
            f"{solution} {platforms}",
        )
    )
    has_admin = not (
        admin in {"none", "no", "not needed", "n/a"}
        or bool(re.search(r"\bno admin(?: dashboard| area)?\b", admin))
    )
    integration_count = 0 if _explicit_none(integrations) else _count_items(integrations)

    if (
        platform_count >= 2
        or (native_app and has_admin)
        or features >= 9
        or integration_count >= 4
        or page_count >= 20
    ):
        return "complex"
    if (
        any(marker in solution for marker in ("marketing website", "multi page", "multi-page", "portfolio"))
        and not native_app
        and not has_admin
        and integration_count <= 1
        and (page_count == 0 or page_count <= 8)
        and features <= 6
    ):
        return "small"
    if (
        "website" in solution
        and "web app" not in solution
        and not native_app
        and not has_admin
        and integration_count <= 1
        and features <= 6
        and (page_count == 0 or page_count <= 8)
    ):
        return "small"
    return "standard"


def _explicit_none(value: str) -> bool:
    return value in {"", "none", "no", "not needed", "n/a"} or bool(
        re.search(r"\bno integrations?\b", value)
    )


def _scope_count(value: str) -> int:
    numeric = re.search(r"\b(\d+)\s+(?:pages?|screens?|sections?)\b", value)
    if numeric:
        return int(numeric.group(1))
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    written = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:pages?|screens?|sections?)\b",
        value,
    )
    return words.get(written.group(1), 0) if written else 0


def _normalized_scope_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    elif isinstance(value, dict):
        value = " ".join(f"{key} {item}" for key, item in value.items())
    return " ".join(str(value or "").lower().split())


def _fallback_pricing_signals(request: ProjectQuoteRequest) -> List[str]:
    brief = request.brief or {}
    return [
        f"{_count_items(brief.get('coreFeatures')) or 'Several'} core feature area(s) in scope.",
        f"{max(1, _count_items(brief.get('platforms')))} platform target(s) included.",
        f"Scope classified as {_scope_tier(brief)} using confirmed pages, workflows, platforms, integrations, and admin needs.",
        "Fixed-price freelance marketplace benchmarks were used instead of salary data.",
        "Proportionate architecture and UI/UX planning included before implementation.",
    ]


def _default_assumptions() -> List[str]:
    return [
        "The first release follows the confirmed brief without major scope expansion.",
        "The final escrow amount funds planning and implementation for the agreed fixed scope.",
        "The delivery team accepts the project before the customer can fund escrow.",
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


def _currency_per_usd(currency: str) -> float:
    if currency == "USD":
        return 1.0
    configured: Dict[str, float] = {}
    for entry in os.getenv("CURRENCY_RATES_PER_USD", "").split(","):
        code, separator, raw_rate = entry.partition(":")
        if not separator:
            continue
        try:
            rate = float(raw_rate)
        except ValueError:
            continue
        if code.strip() and rate > 0:
            configured[code.strip().upper()] = rate
    return configured.get(currency, {"EGP": 48.5, "EUR": 0.92, "GBP": 0.79}.get(currency, 1.0))


def _fixed_package_total(request: ProjectQuoteRequest, tier: str) -> float:
    brief = request.brief or {}
    feature_count = _count_items(brief.get("coreFeatures"))
    platform_count = max(1, _count_items(brief.get("platforms")))
    page_count = _scope_count(_normalized_scope_text(brief.get("scopeDetails")))
    integrations = _normalized_scope_text(brief.get("integrations"))
    integration_count = 0 if _explicit_none(integrations) else _count_items(brief.get("integrations"))
    scope_text = " ".join(
        _normalized_scope_text(brief.get(field))
        for field in ("solutionType", "platforms", "scopeDetails", "adminNeeds")
    )
    native_app = bool(
        re.search(r"\b(?:mobile app|native app|ios|android|flutter|react native)\b", scope_text)
    )
    admin = _normalized_scope_text(brief.get("adminNeeds"))
    has_admin = not (
        admin in {"", "none", "no", "not needed", "n/a"}
        or bool(re.search(r"\bno admin(?: dashboard| area)?\b", admin))
    )
    variable = (
        max(feature_count - 1, 0)
        * {"trivial": 4, "small": 10, "standard": 28, "complex": 55}[tier]
        + max(page_count - (1 if tier == "trivial" else 3), 0)
        * {"trivial": 3, "small": 12, "standard": 24, "complex": 38}[tier]
        + integration_count
        * {"trivial": 0, "small": 25, "standard": 65, "complex": 110}[tier]
        + ({"trivial": 0, "small": 50, "standard": 130, "complex": 220}[tier] if has_admin else 0)
        + ((350 if tier == "complex" else 190) if native_app else 0)
        + max(platform_count - 1, 0) * (220 if tier == "complex" else 90)
    )
    deadline_multiplier = 1 + min(_deadline_pressure(request.project.get("deadline")), 0.12)
    return _round_money(
        (_TIER_BASE_USD[tier] + variable)
        * deadline_multiplier
        * _currency_per_usd(_currency(request.project))
    )


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
        return len([item for item in re.split(r",|;|\n", value) if item.strip()])
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


def _fixed_role_estimate(
    role_key: str,
    people: int,
    hours_each: float,
    package_amount: float,
) -> RoleEstimate:
    subtotal = _round_money(package_amount)
    effective_rate = _round_money(subtotal / (people * hours_each))
    return RoleEstimate(
        roleKey=role_key,
        people=people,
        hoursEach=hours_each,
        hourlyRate=effective_rate,
        subtotal=subtotal,
    )


def _ceil_div(value: float, divisor: int) -> int:
    return int((value + divisor - 1) // divisor)
