import json
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
GENAI_TIMEOUT = 60.0


class RoleBriefGenerationError(RuntimeError):
    """Raised when role brief generation cannot produce valid structured data."""


class RoleBriefRequest(BaseModel):
    assignmentId: str
    roleKey: str
    project: Dict[str, Any] = Field(default_factory=dict)
    brief: Optional[Dict[str, Any]] = Field(default_factory=dict)
    standardExpectations: List[str] = Field(default_factory=list)
    freelancer: Optional[Dict[str, Any]] = None


class RoleBriefResponse(BaseModel):
    title: str
    summary: str
    objectives: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    requiredInputs: List[str] = Field(default_factory=list)
    expectedDeliverables: List[str] = Field(default_factory=list)
    acceptanceCriteria: List[str] = Field(default_factory=list)
    handoffChecklist: List[str] = Field(default_factory=list)
    collaborationNotes: str
    suggestedQuestions: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)


def generate_role_brief(request: RoleBriefRequest) -> Dict[str, Any]:
    prompt = _build_prompt(request)
    client = genai.Client()

    try:
        response = _generate_response(client, prompt)
        if not response.text:
            raise RoleBriefGenerationError("Empty response from AI.")

        result = json.loads(response.text)
        validated = RoleBriefResponse(**result)
        return validated.model_dump()
    except RoleBriefGenerationError:
        raise
    except ValidationError as exc:
        logger.exception("Role brief response validation failed")
        raise RoleBriefGenerationError(
            f"Role brief response validation failed: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        logger.exception("Role brief response was not valid JSON")
        raise RoleBriefGenerationError(
            "AI response could not be parsed as JSON."
        ) from exc
    except errors.APIError as exc:
        logger.exception("Gemini role brief generation request failed")
        raise RoleBriefGenerationError(
            "AI provider is temporarily unavailable. Please retry shortly."
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected role brief generation error")
        raise RoleBriefGenerationError("Failed to generate role brief.") from exc


def _build_prompt(request: RoleBriefRequest) -> str:
    input_json = json.dumps(
        {
            "assignmentId": request.assignmentId,
            "roleKey": request.roleKey,
            "project": request.project,
            "brief": request.brief or {},
            "standardExpectations": request.standardExpectations,
            "freelancer": request.freelancer,
        },
        indent=2,
        default=str,
    )
    schema_json = json.dumps(RoleBriefResponse.model_json_schema(), indent=2)
    role_label = "UI/UX designer" if request.roleKey == "ui_ux" else "software architect"

    return f"""
You are the Nexus planning-assignment brief agent.

You create a clear, project-specific brief for the assigned {role_label}. The
backend already provides standard expectations for the role. Your job is to
adapt those rules to the actual project so the freelancer knows exactly what to
produce before the Scrum Master creates implementation milestones.

Input:
{input_json}

Rules:
- Use the confirmed project and brief details. Do not invent unrelated scope.
- Be specific enough that the freelancer can start work immediately.
- Mention project-specific domain, users, core features, platforms, budget,
  deadline, constraints, and open questions when available.
- Keep customer-facing language warm and plain; keep delivery criteria precise.
- If the brief is missing something important, put it in suggestedQuestions
  instead of pretending it is known.
- For architecture roles, focus on stack, modules, APIs, data model, security,
  integrations, performance, deployment, and implementation risks.
- For UI/UX roles, focus on user journeys, screen map, components, responsive
  behavior, accessibility, visual direction, states, and handoff details.
- Return JSON only. No markdown. No extra text.
- Match this exact output schema:
{schema_json}
"""


def _get_model_candidates() -> List[str]:
    primary = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
    models = [primary] + [m.strip() for m in fallbacks if m.strip()]
    return list(dict.fromkeys(models))


def _generate_response(client, prompt_text: str):
    models = _get_model_candidates()
    if not models:
        raise RoleBriefGenerationError("No Gemini model configured.")

    last_model = models[-1]
    for model in models:
        try:
            return client.models.generate_content(
                model=model,
                contents=[prompt_text],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    top_k=1,
                    top_p=0.1,
                    http_options=types.HttpOptions(timeout=int(GENAI_TIMEOUT * 1000)),
                ),
            )
        except errors.APIError as exc:
            if model == last_model:
                raise
            logger.warning(
                "Gemini role brief generation failed with model '%s'; trying fallback: %s",
                model,
                exc,
            )

    raise RoleBriefGenerationError("All Gemini models failed.")
