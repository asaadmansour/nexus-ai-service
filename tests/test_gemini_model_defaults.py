import unittest

from app.agents.assessment_generation import DEFAULT_ASSESSMENT_GEMINI_MODEL
from app.agents.assessment_grading import DEFAULT_GRADING_GEMINI_MODEL
from app.agents.cv_extraction import DEFAULT_CV_GEMINI_MODEL
from app.agents.planning_submission_evaluation import (
    DEFAULT_GEMINI_MODEL as PLANNING_EVALUATION_MODEL,
)
from app.agents.project_plan_generation import (
    DEFAULT_GEMINI_MODEL as PROJECT_PLAN_MODEL,
)
from app.agents.project_quote_estimation import (
    DEFAULT_GEMINI_MODEL as PROJECT_QUOTE_MODEL,
)
from app.agents.requirements.llm import DEFAULT_GEMINI_MODEL as REQUIREMENTS_MODEL
from app.agents.role_brief_generation import (
    DEFAULT_GEMINI_MODEL as ROLE_BRIEF_MODEL,
)
from app.agents.submission_evaluation import (
    DEFAULT_GEMINI_MODEL as SUBMISSION_EVALUATION_MODEL,
)


class GeminiModelDefaultsTests(unittest.TestCase):
    def test_all_generative_agents_share_the_supported_default(self):
        expected = "gemini-3.1-flash-lite"
        self.assertEqual(
            {
                DEFAULT_ASSESSMENT_GEMINI_MODEL,
                DEFAULT_GRADING_GEMINI_MODEL,
                DEFAULT_CV_GEMINI_MODEL,
                PLANNING_EVALUATION_MODEL,
                PROJECT_PLAN_MODEL,
                PROJECT_QUOTE_MODEL,
                REQUIREMENTS_MODEL,
                ROLE_BRIEF_MODEL,
                SUBMISSION_EVALUATION_MODEL,
            },
            {expected},
        )


if __name__ == "__main__":
    unittest.main()
