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

# Evidence types the model cannot verify on its own (no repo clone / render),
# so a positive AI result on these should still be confirmed by a human.
HUMAN_REVIEW_EVIDENCE_TYPES = {
    "pdf",
    "repo",
    "pull_request",
    "figma",
    "zip",
    "other",
}


class SubmissionEvaluationError(RuntimeError):
    """Raised when the AI provider or validation fails."""


class RubricItem(BaseModel):
    criterion: str
    met: bool
    evidence: str


class SubmissionEvaluationResponse(BaseModel):
    passed: bool
    score: int  # 0-100
    revisionRequested: bool
    revisionNotes: str
    requiresHumanReview: bool
    rubric: List[RubricItem] = Field(default_factory=list)


def evaluate_submission(request: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_prompt(request)
    client = genai.Client()

    try:
        response = _generate_evaluation_response(client, prompt)
        if not response.text:
            raise SubmissionEvaluationError("Empty response from AI.")
        result = json.loads(response.text)
        validated = SubmissionEvaluationResponse(**result)
        return _normalize(validated, request)

    except SubmissionEvaluationError:
        raise
    except errors.APIError as e:
        logger.exception("Gemini submission evaluation request failed")
        raise SubmissionEvaluationError(
            "AI provider is temporarily unavailable. Please retry shortly."
        ) from e
    except json.JSONDecodeError as e:
        logger.exception("LLM response is not valid JSON")
        raise SubmissionEvaluationError(
            "AI response could not be parsed as JSON. Please try again."
        ) from e
    except ValidationError as e:
        logger.exception("LLM response failed schema validation")
        raise SubmissionEvaluationError(
            f"AI response validation failed: {e}"
        ) from e
    except Exception as e:
        logger.exception("Unexpected error in submission evaluation")
        raise SubmissionEvaluationError(
            "Failed to evaluate submission using AI."
        ) from e


def _normalize(
    validated: SubmissionEvaluationResponse, request: Dict[str, Any]
) -> Dict[str, Any]:
    """Clamp/derive fields so the backend can trust the shape regardless of model drift."""
    data = validated.model_dump()
    data["score"] = max(0, min(100, int(data.get("score", 0))))

    submission = request.get("submission") or {}
    submission_type = str(submission.get("submissionType", "other")).lower()

    # Evidence the model cannot verify itself always needs a human confirm.
    if submission_type in HUMAN_REVIEW_EVIDENCE_TYPES:
        data["requiresHumanReview"] = True

    task = request.get("task") or {}
    criteria = _dedupe_strings(
        list(task.get("acceptanceCriteria") or [])
        + list(task.get("deliverables") or [])
    )
    returned = {
        str(item.get("criterion", "")).strip(): item
        for item in data.get("rubric", [])
        if str(item.get("criterion", "")).strip()
    }
    if criteria:
        data["rubric"] = [
            returned.get(criterion)
            or {
                "criterion": criterion,
                "met": False,
                "evidence": "The evaluation returned no evidence for this criterion.",
            }
            for criterion in criteria
        ]
    else:
        data["rubric"] = [
            {
                "criterion": "Task acceptance criteria are defined",
                "met": False,
                "evidence": "The task has no acceptance criteria or deliverables to evaluate.",
            }
        ]
        data["requiresHumanReview"] = True

    unmet = [item for item in data["rubric"] if not item.get("met")]
    if unmet:
        data["passed"] = False
        data["score"] = min(data["score"], 69)

    # A failed evaluation must request a revision; a pass must not.
    if not data["passed"]:
        data["revisionRequested"] = True
    else:
        data["revisionRequested"] = False

    if data["revisionRequested"] and not data.get("revisionNotes", "").strip():
        data["revisionNotes"] = (
            "One or more acceptance criteria are not fully met. Address the "
            "unmet rubric items and resubmit."
        )
    return data


def _dedupe_strings(values: List[Any]) -> List[str]:
    return list(
        dict.fromkeys(str(value).strip() for value in values if str(value).strip())
    )


def _build_prompt(request: Dict[str, Any]) -> str:
    input_json = json.dumps(request, indent=2)
    schema_dict = SubmissionEvaluationResponse.model_json_schema()
    schema_json = json.dumps(schema_dict, indent=2)

    prompt = f"""
You are a senior technical reviewer evaluating a freelancer's DELIVERY submission
for one specific project task. Judge only whether the submitted work satisfies the
task's deliverables and acceptance criteria.

Input:
{input_json}

Evidence handling by submissionType:
- "text": read submissionText directly.
- "repo" / "pull_request": you cannot clone or run the code. Reason from the
  provided urls, commitSha, notes, and submissionText. If you cannot actually
  verify the code behind a URL, mark the affected rubric items as not met and set
  requiresHumanReview to true.
- "pdf" / "figma" / "zip": you cannot open the file. Evaluate from notes/text only
  and set requiresHumanReview to true.

For every acceptance criterion and deliverable produce one rubric entry:
- criterion: copy the input text exactly
- met: true only if there is concrete evidence it is satisfied
- evidence: one short sentence citing the evidence (or why it is unverifiable)

Then decide:
- passed: true only if all critical criteria are met with real evidence
- score: 0-100 reflecting how complete and correct the work is
- revisionRequested: true when the freelancer should fix and resubmit
- revisionNotes: specific, actionable feedback naming the unmet items (not generic)
- requiresHumanReview: true when you could not fully verify the evidence yourself

Do not invent evidence you were not given. Be strict but fair.
Return strictly valid JSON matching this schema:
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
        raise SubmissionEvaluationError("No Gemini model configured.")

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
                "Gemini submission evaluation failed with model '%s'; trying fallback: %s",
                model,
                exc,
            )
    raise SubmissionEvaluationError("All Gemini models failed.")
