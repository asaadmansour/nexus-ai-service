import json
import logging
from typing import List, Literal

from pydantic import BaseModel
from google import genai
from google.genai import types


logger = logging.getLogger(__name__)


class QuestionResultResponse(BaseModel):
    questionId: str
    score: float
    maxScore: float
    feedback: str


class GradeAssessmentResponse(BaseModel):
    assessmentId: str

    score: float
    maxScore: float

    recommendation: Literal[
        "pass",
        "needs_review",
        "fail"
    ]

    feedback: str

    graderConfidence: float

    questionResults: List[QuestionResultResponse]


class RubricInput(BaseModel):
    maxScore: float
    gradingNotes: str


class QuestionInput(BaseModel):
    id: str
    questionType: str
    skill: str
    difficulty: str
    prompt: str
    rubric: RubricInput


class AnswerValue(BaseModel):
    value: str | None = None


class AnswerInput(BaseModel):
    questionId: str
    answer: AnswerValue



def grade_assessment(
    assessment_id: str,
    questions: List[QuestionInput],
    answers: List[AnswerInput]
) -> dict:


    if not questions:
        raise ValueError(
            "Questions are required for grading."
        )


    if not answers:
        raise ValueError(
            "Answers are required for grading."
        )


    client = genai.Client()


    prompt_text = f"""
You are an expert technical interviewer grading a freelancer assessment.

Assessment ID:
{assessment_id}

You will evaluate:
- Assessment questions
- Hidden grading rubrics
- Candidate answers


IMPORTANT GRADING PRINCIPLES:

- Judge semantic correctness, not exact wording.
- Accept different valid approaches when they demonstrate correct reasoning.
- Do not require the exact technology, library, or terminology mentioned in the rubric unless it is essential.
- Do not penalize grammar unless it prevents understanding.
- Evaluate practical understanding, not memorized definitions.
- Do not give high scores only because the answer contains expected keywords.
- A strong answer should explain why, trade-offs, or practical implementation details when required.
- Consider the candidate's experience level.
- Only evaluate the skills directly tested in this assessment.
- Do not generalize the candidate's entire expertise from this assessment alone.


QUESTION TYPE SPECIFIC GRADING:


Multiple Choice Questions:

- Grade objectively based on the selected option.
- If the selected answer is incorrect, score according to the rubric.
- Do not give partial credit unless the assessment explicitly allows it.


Code Analysis / Debugging Questions:

Evaluate:
- Does the candidate correctly identify the problem?
- Do they explain the underlying reason?
- Do they propose a technically valid solution?
- Do they consider side effects, performance, security, or maintainability when relevant.

Do not require:
- identical code
- identical variable names
- one specific implementation


Scenario Questions:

Evaluate:
- Problem understanding.
- Quality of reasoning.
- Technical decisions.
- Awareness of trade-offs.
- Real-world engineering thinking.

Accept multiple valid solutions if they solve the problem effectively.

Do not mark an answer wrong only because it differs from the rubric's example solution.


PRACTICAL EXPERIENCE SIGNALS:

Higher score indicators:
- References implementation details.
- Explains decisions and trade-offs.
- Considers real production problems.
- Connects concepts to practical usage.

Lower score indicators:
- Only gives textbook definitions.
- Uses vague statements without explaining how.
- Mentions technologies without demonstrating understanding.
- Avoids the actual scenario.


QUESTION SCORING:

For each question:
- Compare the answer against the rubric.
- Assign a score between 0 and maxScore.
- Provide short actionable feedback.
- Explain missing important points when score is reduced.


RECOMMENDATION RULES:

Return only one:

pass:
- Score >= 80%.
- Candidate demonstrates strong understanding of the evaluated topics.

needs_review:
- Score between 50% and 79%.
- Candidate shows partial understanding, inconsistent answers, or unclear practical ability.

fail:
- Score below 50%.
- Candidate shows major misunderstandings, missing answers, or inability to apply concepts.


CONFIDENCE:

Provide graderConfidence between 0.0 and 1.0.

Higher confidence:
- Clear answers.
- Objective questions.
- Strong rubric alignment.

Lower confidence:
- Ambiguous answers.
- Subjective scenarios.
- Insufficient information.


Questions and answers:

"""


    for index, question in enumerate(questions):

        matching_answer = next(
            (
                answer
                for answer in answers
                if answer.questionId == question.id
            ),
            None
        )


        candidate_answer = (
            matching_answer.answer.value
            if matching_answer
            and matching_answer.answer.value
            else "No answer provided"
        )


        prompt_text += f"""

--------------------------------

Question {index + 1}

Question ID:
{question.id}


Skill:
{question.skill}


Difficulty:
{question.difficulty}


Question:
{question.prompt}


Rubric:

Maximum Score:
{question.rubric.maxScore}


Grading Notes:
{question.rubric.gradingNotes}


Candidate Answer:

{candidate_answer}

"""


    prompt_text += """

Return ONLY JSON matching the required schema.
"""


    try:

        response = client.models.generate_content(

            model="gemini-3.5-flash",

            contents=[
                prompt_text
            ],

            config=types.GenerateContentConfig(

                response_mime_type="application/json",

                response_schema=GradeAssessmentResponse,

                temperature=0.0,

                top_k=1,

                top_p=0.1,
            ),
        )


        return json.loads(
            response.text
        )


    except Exception as e:

        logger.exception(
            "LLM assessment grading failed."
        )

        raise ValueError(
            "Failed to grade assessment using AI."
        ) from e