import json
import logging
from typing import Optional, List

from pydantic import BaseModel, Field
from google import genai
from google.genai import types


logger = logging.getLogger(__name__)


class AssessmentChoice(BaseModel):
    id: str
    label: str


class AssessmentRubric(BaseModel):
    correctChoiceId: Optional[str] = None
    maxScore: int
    gradingNotes: str


class AssessmentQuestion(BaseModel):
    questionType: str
    skill: str
    difficulty: str
    prompt: str
    choices: Optional[List[AssessmentChoice]] = None
    rubric: AssessmentRubric
    orderIndex: int


class AssessmentGenerationResponse(BaseModel):
    durationSeconds: int
    questions: List[AssessmentQuestion]


def generate_assessment(
    skills: List[str],
    years_experience: Optional[int],
    headline: Optional[str],
    question_count: int = 6,
    duration_seconds: int = 1800
) -> dict:


    if not skills:
        raise ValueError(
            "Skills are required to generate assessment."
        )


    client = genai.Client()


    prompt_text = f"""
You are an expert technical interviewer creating a challenging freelancer assessment.

Your goal is NOT to test memorized definitions.
Your goal is to evaluate whether the candidate can actually build, debug, maintain, and reason about real software systems.

Candidate profile:
- Headline: {headline or "Unknown"}
- Main skills to evaluate: {', '.join(skills) if skills else "General software development"}
- Years of experience: {years_experience or "Unspecified"}

Assessment requirements:
- Generate EXACTLY {question_count} questions.
- Duration target: {duration_seconds} seconds.
- Questions must match the candidate's experience level.
- Do not ask unrealistic senior-level questions for junior candidates.
- Include at least one scenario question.

Question design rules:

1. Avoid generic theory questions.
DO NOT create questions like:
- "What is React?"
- "What is dependency injection?"
- "Explain JWT."
- "What is Docker?"

These are too easy to answer using AI tools.

Instead, create questions that require reasoning:
- debugging
- architecture decisions
- code analysis
- system design
- trade-off decisions
- explaining existing implementations
- identifying bugs or risks


2. Prefer practical code-based questions.

Include realistic code snippets whenever possible.

Examples:

- Show a React component with a performance issue and ask the candidate to identify the problem.
- Show a backend endpoint with security/performance issues and ask what should change.
- Show a database query and ask how they would optimize it.
- Show a Docker configuration and ask what is wrong.
- Show an API implementation and ask about scalability or security concerns.

For multiple choice questions:
- The question should contain a code snippet or real scenario.
- Options should represent realistic engineering decisions.
- Avoid obvious answers.
- At least two options should look technically reasonable.


3. Evaluate depth in the candidate's claimed skills.

If the candidate claims React:
Ask about:
- component design
- state management decisions
- performance issues
- debugging
- hooks behavior
- architecture choices

If the candidate claims backend skills:
Ask about:
- API design
- authentication
- database decisions
- error handling
- scalability
- security

If the candidate claims DevOps exposure:
Do not test deep DevOps specialization.
Test practical awareness:
- deployment issues
- environment variables
- logging
- monitoring
- Docker basics
- CI/CD concepts
- production debugging


4. Include real-world engineering situations.

Examples:

Instead of:
"What is caching?"

Ask:
"Your API response time increased from 200ms to 5 seconds after adding a new feature. The database CPU is high. Explain how you would investigate and fix it."

Instead of:
"What is SQL injection?"

Ask:
"Review this login endpoint. Identify security risks and explain how you would improve it."


5. Mix question types:

Allowed types:
- multiple_choice
- short_answer
- scenario

Recommended distribution:
- 30-40% code/debugging multiple choice
- 30-40% short answer reasoning questions
- 20-30% scenario questions

6. Make questions resistant to AI-only answering.

Questions should require:
- explaining why
- comparing alternatives
- predicting behavior
- analyzing provided context

Avoid questions where one sentence is enough.


7. Difficulty:

Adjust difficulty based on experience:

0-2 years:
- debugging
- implementation understanding
- common architecture decisions

3-5 years:
- system trade-offs
- optimization
- production scenarios

5+ years:
- architecture
- scalability
- leadership decisions

Scoring distribution:

The total assessment score should be 100 points.

Question weights:
- Multiple choice questions: 10-20 points each.
- Short answer questions: 15-25 points each.
- Scenario questions: 25-40 points each.

Prefer higher scores for:
- scenario questions
- debugging questions
- architecture decisions

Prefer lower scores for:
- direct knowledge questions

The scoring should reflect the difficulty and practical value of each question.


8. Rubric requirements:

Every question MUST include a rubric.

The rubric should explain:
- what a strong answer contains
- important technical points
- common mistakes
- how to score the answer

For multiple choice:
- Include correctChoiceId.

For open questions:
- correctChoiceId should be null.


9. Output rules:

Return ONLY valid JSON matching the provided schema.

Do not include explanations outside JSON.
"""


    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash",

            contents=[
                prompt_text
            ],

            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AssessmentGenerationResponse,
                temperature=0.0,
                top_k=1,
                top_p=0.1,
            ),
        )


        result = json.loads(
            response.text
        )


        return result


    except Exception as e:

        logger.exception(
            "LLM assessment generation failed."
        )

        raise ValueError(
            "Failed to generate assessment using AI."
        ) from e