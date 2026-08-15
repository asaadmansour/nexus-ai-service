import unittest

from app.agents.submission_evaluation import (
    RubricItem,
    SubmissionEvaluationResponse,
    _normalize,
)


class SubmissionEvaluationTests(unittest.TestCase):
    def test_missing_rubric_item_prevents_false_pass(self):
        response = SubmissionEvaluationResponse(
            passed=True,
            score=98,
            revisionRequested=False,
            revisionNotes="",
            requiresHumanReview=False,
            rubric=[
                RubricItem(
                    criterion="API returns the documented response",
                    met=True,
                    evidence="The submitted text includes the response.",
                )
            ],
        )
        request = {
            "task": {
                "acceptanceCriteria": [
                    "API returns the documented response",
                    "Duplicate requests are idempotent",
                ],
                "deliverables": [],
            },
            "submission": {"submissionType": "pull_request"},
        }

        result = _normalize(response, request)

        self.assertFalse(result["passed"])
        self.assertTrue(result["revisionRequested"])
        self.assertLessEqual(result["score"], 69)
        self.assertEqual(len(result["rubric"]), 2)
        self.assertFalse(result["rubric"][1]["met"])
        self.assertTrue(result["requiresHumanReview"])

    def test_text_submission_can_pass_complete_rubric(self):
        response = SubmissionEvaluationResponse(
            passed=True,
            score=90,
            revisionRequested=False,
            revisionNotes="",
            requiresHumanReview=False,
            rubric=[RubricItem(criterion="Explain the algorithm", met=True, evidence="Done")],
        )
        request = {
            "task": {"acceptanceCriteria": ["Explain the algorithm"]},
            "submission": {"submissionType": "text"},
        }

        result = _normalize(response, request)

        self.assertTrue(result["passed"])
        self.assertFalse(result["revisionRequested"])
        self.assertFalse(result["requiresHumanReview"])


if __name__ == "__main__":
    unittest.main()
