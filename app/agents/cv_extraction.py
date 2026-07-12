import ipaddress
import socket
import urllib.request
import logging
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field
from google import genai
from google.genai import types


logger = logging.getLogger(__name__)


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
READ_CHUNK = MAX_FILE_SIZE + 1      # read one byte over the limit to detect oversize


class CVSummary(BaseModel):
    education: Optional[str] = None
    experience: Optional[str] = None
    projects: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)


class CVExtractionResponse(BaseModel):
    headline: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    yearsExperience: Optional[int] = None
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

            content_type = response.headers.get(
                "Content-Type",
                ""
            )

            if "pdf" not in content_type.lower():
                raise ValueError(
                    "Provided URL is not a PDF file."
                )

            # Bound the read before checking size to avoid loading unlimited data
            pdf_bytes = response.read(READ_CHUNK)

            if len(pdf_bytes) > MAX_FILE_SIZE:
                raise ValueError(
                    "CV file is too large. Maximum size is 10MB."
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

        response = client.models.generate_content(
            model="gemini-3.5-flash",
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

        # Prefer SDK-validated parsed result, fall back to JSON text
        if response.parsed is not None:
            result = response.parsed.model_dump()
        else:
            import json
            result = json.loads(response.text)

        result["cvUrl"] = cv_url
        result["source"] = "llm"

        return result

    except Exception as e:
        logger.exception(
            "LLM extraction failed."
        )

        raise ValueError(
            "Failed to extract CV using AI."
        ) from e