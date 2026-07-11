import urllib.request
import json
import logging
from typing import Optional

from pydantic import BaseModel, Field
from google import genai
from google.genai import types


logger = logging.getLogger(__name__)


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


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


def process_cv_with_llm(cv_url: str) -> dict:

    # Download CV PDF
    try:
        req = urllib.request.Request(
            cv_url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(
            req,
            timeout=15
        ) as response:

            content_type = response.headers.get(
                "Content-Type",
                ""
            )

            if "pdf" not in content_type.lower():
                raise ValueError(
                    "Provided URL is not a PDF file."
                )

            pdf_bytes = response.read()

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
                temperature=0.0,
                top_k=1,
                top_p=0.1,
            ),
        )


        result_json = json.loads(
            response.text
        )

        result_json["cvUrl"] = cv_url
        result_json["source"] = "llm"


        return result_json


    except Exception as e:
        logger.exception(
            "LLM extraction failed."
        )

        raise ValueError(
            "Failed to extract CV using AI."
        ) from e