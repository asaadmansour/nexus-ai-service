import json
import logging
import os
from typing import Any, Dict, List, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
GENAI_TIMEOUT = 60.0


class PlanningSubmissionEvaluationError(RuntimeError):
    """Raised when the AI provider or response validation fails."""


class RequirementCheck(BaseModel):
    key: str
    title: str
    status: Literal["met", "partial", "missing", "conflict"]
    mandatory: bool
    severity: Literal["info", "minor", "major", "blocker"]
    evidence: str
    feedback: str


class EvaluateSubmissionResponse(BaseModel):
    passed: bool
    score: float = Field(ge=0, le=100)
    recommendation: Literal["approve", "changes_requested", "reject"]
    summary: str
    checks: List[RequirementCheck] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    revisionItems: List[str] = Field(default_factory=list)
    crossContractIssues: List[str] = Field(default_factory=list)


def evaluate_submission(request: Dict[str, Any]) -> Dict[str, Any]:
    _validate_request_contract(request)
    prompt = _build_prompt(request)
    client = genai.Client()

    try:
        response = _generate_evaluation_response(client, prompt)
        if not response.text:
            raise PlanningSubmissionEvaluationError("Empty response from AI.")
        raw_result = json.loads(response.text)
        return _normalize_evaluation(request, raw_result).model_dump()
    except PlanningSubmissionEvaluationError:
        raise
    except errors.APIError as exc:
        logger.exception("Gemini planning evaluation request failed")
        raise PlanningSubmissionEvaluationError(
            "AI provider is temporarily unavailable. Please retry shortly."
        ) from exc
    except json.JSONDecodeError as exc:
        logger.exception("Planning evaluation response is not valid JSON")
        raise PlanningSubmissionEvaluationError(
            "AI response could not be parsed as JSON. Please try again."
        ) from exc
    except ValidationError as exc:
        logger.exception("Planning evaluation response failed schema validation")
        raise PlanningSubmissionEvaluationError(
            f"AI response validation failed: {exc}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected planning submission evaluation error")
        raise PlanningSubmissionEvaluationError(
            "Failed to evaluate the planning submission using AI."
        ) from exc


def _validate_request_contract(request: Dict[str, Any]) -> None:
    submission = request.get("submission") or {}
    submission_type = submission.get("submissionType")
    requirements = request.get("requirements") or []
    if submission_type not in {"architecture", "ui_ux"}:
        raise ValueError("submissionType must be architecture or ui_ux")
    if not requirements:
        raise ValueError("At least one planning requirement is required")
    if submission_type == "ui_ux" and not request.get("approvedArchitecture"):
        raise ValueError("UI/UX evaluation requires the approved architecture")


def _normalize_evaluation(
    request: Dict[str, Any], raw_result: Dict[str, Any]
) -> EvaluateSubmissionResponse:
    returned_checks = {
        item.get("key"): item
        for item in raw_result.get("checks", [])
        if isinstance(item, dict) and item.get("key")
    }
    content = (request.get("submission") or {}).get("content") or {}
    evidence_map = content.get("requirementEvidence") or {}
    checks: List[RequirementCheck] = []

    for requirement in request.get("requirements", []):
        key = requirement["key"]
        candidate = returned_checks.get(key) or {}
        evidence = evidence_map.get(key) or {}
        summary = str(evidence.get("summary") or "").strip()
        urls = _string_list(evidence.get("urls"))
        has_evidence = bool(summary or urls)
        has_required_url = not requirement.get("requiresUrl") or bool(urls)
        requested_status = candidate.get("status")
        valid_statuses = {"met", "partial", "missing", "conflict"}
        status = (
            requested_status
            if has_evidence
            and has_required_url
            and requested_status in valid_statuses
            else "missing"
        )
        mandatory = bool(requirement.get("mandatory", True))
        severity = candidate.get("severity")
        if severity not in {"info", "minor", "major", "blocker"}:
            severity = (
                "info" if status == "met" else "blocker" if mandatory else "minor"
            )
        if mandatory and status != "met":
            severity = "blocker"

        feedback = str(candidate.get("feedback") or "").strip()
        if status != "met" and not feedback:
            feedback = f"Complete {requirement['title']} with project-specific details"
            feedback += (
                " and an accessible evidence URL."
                if requirement.get("requiresUrl")
                else "."
            )
        checks.append(
            RequirementCheck(
                key=key,
                title=requirement["title"],
                status=status,
                mandatory=mandatory,
                severity=severity,
                evidence=str(
                    candidate.get("evidence")
                    or summary
                    or ", ".join(urls)
                    or "No evidence submitted."
                ),
                feedback=feedback
                or "The submitted evidence satisfies this requirement.",
            )
        )

    blockers = [
        check for check in checks if check.mandatory and check.status != "met"
    ]
    raw_score = _bounded_score(raw_result.get("score"), checks)
    score = min(raw_score, 69.0) if blockers else raw_score
    requested_recommendation = raw_result.get("recommendation")
    if blockers or score < 80:
        recommendation = "changes_requested"
    elif requested_recommendation == "reject":
        recommendation = "reject"
    elif requested_recommendation == "changes_requested":
        recommendation = "changes_requested"
    else:
        recommendation = "approve"

    revisions = _dedupe(
        _string_list(raw_result.get("revisionItems"))
        + [f"{check.title}: {check.feedback}" for check in blockers]
    )
    summary = str(raw_result.get("summary") or "").strip() or (
        "All mandatory planning requirements are complete and consistent."
        if recommendation == "approve"
        else f"Revision is required for {len(blockers)} mandatory requirement(s)."
    )
    return EvaluateSubmissionResponse(
        passed=recommendation == "approve",
        score=score,
        recommendation=recommendation,
        summary=summary,
        checks=checks,
        strengths=_string_list(raw_result.get("strengths")),
        risks=_string_list(raw_result.get("risks")),
        revisionItems=revisions,
        crossContractIssues=_string_list(raw_result.get("crossContractIssues")),
    )


def _bounded_score(value: Any, checks: List[RequirementCheck]) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        met = sum(1 for check in checks if check.status == "met")
        score = (met / len(checks) * 100) if checks else 0
    return round(max(0, min(100, score)), 2)


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _build_prompt(request: Dict[str, Any]) -> str:
    schema_json = json.dumps(EvaluateSubmissionResponse.model_json_schema(), indent=2)
    input_json = json.dumps(request, indent=2, default=str)
    submission_type = (request.get("submission") or {}).get("submissionType")
    specialist_rules = (
        """
For architecture, verify system context, diagrams, technology decisions, module/data
ownership, complete API/event contracts, data model constraints, auth/security,
integration failure behavior, measurable non-functional requirements, deployment,
observability, testing, and an implementation handoff that lets parallel developers
work without inventing contracts.
"""
        if submission_type == "architecture"
        else """
For UI/UX, verify information architecture, complete user and admin flows, wireframes,
high-fidelity responsive screens, clickable prototype, all loading/empty/error/success/
permission states, accessibility, design system, assets, and an exact screen-to-API/data
mapping. Cross-check every claimed endpoint, field, role, validation, and state against
approvedArchitecture. Put every mismatch in crossContractIssues and mark its relevant
requirement conflict.
"""
    )
    return f"""
You are the strict Nexus AI planning quality gate. Evaluate evidence, not promises.
{specialist_rules}

Rules:
- Return exactly one check for every input requirement, using its exact key and title.
- "met" means the evidence is project-specific, complete, internally consistent, and
  sufficient for another freelancer to implement independently.
- Use "partial" when details exist but are incomplete, "missing" when absent or too
  vague, and "conflict" when inconsistent with the brief or approved architecture.
- Every mandatory partial/missing/conflict is a blocker and requires revision.
- Do not approve with any blocker or score below 80.
- Feedback and revisionItems must say exactly what artifact or contract detail to add.
- URLs are evidence pointers only; do not claim you opened or verified their contents.
- The admin remains the final approver; this output is a recommendation.

Input:
{input_json}

Return only JSON matching this schema:
{schema_json}
"""


def _get_model_candidates() -> List[str]:
    primary = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
    models = [primary] + [model.strip() for model in fallbacks if model.strip()]
    return list(dict.fromkeys(models))


def _generate_evaluation_response(client, prompt_text: str):
    models = _get_model_candidates()
    if not models:
        raise PlanningSubmissionEvaluationError("No Gemini model configured.")

    last_model = models[-1]
    for model in models:
        try:
            return client.models.generate_content(
                model=model,
                contents=[prompt_text],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EvaluateSubmissionResponse,
                    temperature=0.1,
                    top_k=1,
                    top_p=0.1,
                    http_options=types.HttpOptions(
                        timeout=int(GENAI_TIMEOUT * 1000)
                    ),
                ),
            )
        except errors.APIError as exc:
            if model == last_model:
                raise
            logger.warning(
                "Gemini planning evaluation failed with model '%s'; trying fallback: %s",
                model,
                exc,
            )
    raise PlanningSubmissionEvaluationError("All Gemini models failed.")
