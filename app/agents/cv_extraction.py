import ipaddress
import os
import socket
import urllib.request
import logging
from typing import Literal, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import errors, types


load_dotenv()

logger = logging.getLogger(__name__)


DEFAULT_CV_GEMINI_MODEL = "gemini-3.1-flash-lite"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
READ_CHUNK = MAX_FILE_SIZE + 1      # read one byte over the limit to detect oversize


class CVExtractionServiceError(RuntimeError):
    """Raised when the AI extraction provider cannot complete the request."""


class CVSummary(BaseModel):
    education: Optional[str] = None
    experience: Optional[str] = None
    projects: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)


class CVExtractionResponse(BaseModel):
    headline: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    yearsExperience: Optional[int] = None
    professionalRole: Optional[Literal[
        "backend", "frontend", "fullstack", "mobile", "ui_ux",
        "qa", "devops", "data", "ai_ml", "architect"
    ]] = None
    seniorityLevel: Optional[Literal["junior", "mid", "senior"]] = None
    summary: CVSummary = Field(default_factory=CVSummary)
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )


def _assert_public_host(hostname: str) -> None:
    """
    Resolve hostname and reject loopback, private, link-local,
    multicast, reserved, or any other non-public address.
    Raises ValueError on any rejected destination.
    """
    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(
            f"Cannot resolve hostname: {hostname}"
        ) from exc

    for _, _, _, _, sockaddr in addresses:
        raw_ip = sockaddr[0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError as exc:
            raise ValueError(
                f"Invalid IP address returned for hostname: {raw_ip}"
            ) from exc

        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError(
                f"CV URL resolves to a disallowed address: {raw_ip}"
            )


def _validate_url(url: str) -> None:
    """
    Parse and SSRF-validate a URL before fetching it.
    Allows only HTTP and HTTPS schemes with a public destination.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            "CV URL must use HTTP or HTTPS."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(
            "CV URL must contain a valid hostname."
        )

    _assert_public_host(hostname)


def _get_model_candidates() -> list[str]:
    primary_model = (
        os.getenv("GEMINI_CV_MODEL")
        or os.getenv("GEMINI_MODEL")
        or DEFAULT_CV_GEMINI_MODEL
    )
    fallback_models = os.getenv("GEMINI_CV_FALLBACK_MODELS", "")

    models = [
        primary_model,
        *[
            model.strip()
            for model in fallback_models.split(",")
            if model.strip()
        ],
    ]

    return list(dict.fromkeys(models))


def _generate_cv_extraction_response(client, pdf_bytes: bytes, prompt: str):
    models = _get_model_candidates()

    if not models:
        raise CVExtractionServiceError(
            "No Gemini model is configured for CV extraction."
        )

    last_model = models[-1]

    for model in models:
        try:
            return client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(
                        data=pdf_bytes,
                        mime_type="application/pdf"
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CVExtractionResponse,
                ),
            )
        except errors.APIError as exc:
            if model == last_model:
                raise

            logger.warning(
                "Gemini CV extraction failed with model '%s'; trying fallback: %s",
                model,
                exc,
            )


def process_cv_with_llm(cv_url: str) -> dict:

    # Validate URL scheme and destination before fetching (SSRF protection)
    try:
        _validate_url(cv_url)
    except ValueError:
        raise

    # Download CV PDF
    try:
        req = urllib.request.Request(
            cv_url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        # Disable automatic redirects; we revalidate every redirect hop
        opener = urllib.request.build_opener(
            urllib.request.HTTPRedirectHandler()
        )

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                try:
                    _validate_url(newurl)
                except ValueError as exc:
                    raise urllib.error.URLError(
                        f"Redirect blocked (SSRF): {exc}"
                    ) from exc
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        opener = urllib.request.build_opener(_NoRedirect())

        with opener.open(req, timeout=15) as response:

            # Bound the read before checking size to avoid loading unlimited data.
            # Cloudinary raw assets may be served as application/octet-stream or
            # without a .pdf-looking URL, so validate the actual PDF header bytes.
            pdf_bytes = response.read(READ_CHUNK)

            if len(pdf_bytes) > MAX_FILE_SIZE:
                raise ValueError(
                    "CV file is too large. Maximum size is 10MB."
                )

            if b"%PDF-" not in pdf_bytes[:1024]:
                raise ValueError(
                    "Provided URL is not a PDF file."
                )

    except ValueError:
        raise

    except Exception as e:
        logger.exception(
            "Failed to download CV."
        )

        raise ValueError(
            "Failed to retrieve CV file."
        ) from e


    # Send PDF to Gemini
    try:
        client = genai.Client()

        prompt = """
You are an expert CV extraction assistant.

Analyze the provided CV document and extract structured freelancer information.

Rules:
- Extract ONLY information explicitly present in the CV.
- Never invent skills, experience, projects, education, or achievements.
- If information is missing, return null or an empty list.
- Calculate yearsExperience only when employment dates are clearly available.
- Normalize skill names when appropriate while preserving factual meaning.

Extract:

- headline:
  The professional title of the candidate.

- skills:
  Technical and professional skills mentioned in the CV.

- yearsExperience:
  Total professional experience calculated only from available dates.

- professionalRole:
  The single strongest supported professional track, using only one of:
  backend, frontend, fullstack, mobile, ui_ux, qa, devops, data, ai_ml, architect.
  Return null if the CV does not support a clear track.

- seniorityLevel:
  The CV-indicated assessment level: junior, mid, or senior. Use stated title,
  scope, ownership, and clearly supported years of experience. This is only the
  difficulty assigned before assessment, not the candidate's final platform rank.

- summary:
  Include:
  - education
  - experience summary
  - projects
  - strengths

- confidence:
  A value between 0.0 and 1.0 representing confidence in the extraction.

Return JSON only.
"""

        response = _generate_cv_extraction_response(
            client,
            pdf_bytes,
            prompt,
        )

        # Prefer SDK-validated parsed result, fall back to JSON text
        if response.parsed is not None:
            result = response.parsed.model_dump()
        else:
            import json
            result = json.loads(response.text)

        result["cvUrl"] = cv_url
        result["source"] = "llm"

        return result

    except errors.APIError as e:
        logger.exception(
            "Gemini CV extraction request failed."
        )

        raise CVExtractionServiceError(
            "CV extraction AI provider is temporarily unavailable. Please retry shortly."
        ) from e

    except Exception as e:
        logger.exception(
            "LLM extraction failed."
        )

        raise CVExtractionServiceError(
            "Failed to extract CV using AI."
        ) from e
