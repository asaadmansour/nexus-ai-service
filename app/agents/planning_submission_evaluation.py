import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import errors, types

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
GENAI_TIMEOUT = 60.0


class PlanningSubmissionEvaluationError(RuntimeError):
    """Raised when the AI provider or validation fails."""


class EvaluateSubmissionResponse(BaseModel):
    recommendation: str  # "approve", "changes_requested", "reject"
    score: int  # 0-100
    strengths: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    suggestedAdminNotes: str


def evaluate_submission(request: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_prompt(request)
    client = genai.Client()

    try:
        response = _generate_evaluation_response(client, prompt)
        if not response.text:
            raise PlanningSubmissionEvaluationError("Empty response from AI.")
        result = json.loads(response.text)
        validated = EvaluateSubmissionResponse(**result)
        return validated.dict()

    except PlanningSubmissionEvaluationError:
        raise
    except errors.APIError as e:
        logger.exception("Gemini evaluation request failed")
        raise PlanningSubmissionEvaluationError(
            "AI provider is temporarily unavailable. Please retry shortly."
        ) from e
    except json.JSONDecodeError as e:
        logger.exception("LLM response is not valid JSON")
        raise PlanningSubmissionEvaluationError(
            "AI response could not be parsed as JSON. Please try again."
        ) from e
    except ValidationError as e:
        logger.exception("LLM response failed schema validation")
        raise PlanningSubmissionEvaluationError(
            f"AI response validation failed: {e}"
        ) from e
    except Exception as e:
        logger.exception("Unexpected error in submission evaluation")
        raise PlanningSubmissionEvaluationError(
            "Failed to evaluate submission using AI."
        ) from e


def _build_prompt(request: Dict[str, Any]) -> str:
    input_json = json.dumps(request, indent=2)
    schema_dict = EvaluateSubmissionResponse.model_json_schema()
    schema_json = json.dumps(schema_dict, indent=2)

    prompt = f"""
You are an expert project manager reviewing a planning deliverable (architecture or UI/UX).

You are given:
{input_json}

Evaluate the submission and provide:
- recommendation: "approve", "changes_requested", or "reject"
- score: 0-100
- strengths: list of short phrases
- risks: list of short phrases
- suggestedAdminNotes: brief actionable guidance

Return strictly valid JSON matching:
{schema_json}
"""
    return prompt


def _get_model_candidates() -> List[str]:
    primary = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", "").split(",")
    models = [primary] + [m.strip() for m in fallbacks if m.strip()]
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
                "Gemini evaluation failed with model '%s'; trying fallback: %s",
                model,
                exc,
            )
    raise PlanningSubmissionEvaluationError("All Gemini models failed.")