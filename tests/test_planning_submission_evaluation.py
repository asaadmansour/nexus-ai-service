import unittest

from app.agents.planning_submission_evaluation import (
    _normalize_evaluation,
    _validate_request_contract,
)


class PlanningSubmissionEvaluationTests(unittest.TestCase):
    def test_missing_mandatory_artifact_forces_revision(self):
        request = {
            "submission": {
                "submissionType": "architecture",
                "content": {
                    "requirementEvidence": {
                        "system_context": {"summary": "Users and boundaries", "urls": []}
                    }
                },
            },
            "requirements": [
                {
                    "key": "system_context",
                    "title": "System context",
                    "mandatory": True,
                    "requiresUrl": False,
                },
                {
                    "key": "architecture_diagram",
                    "title": "Architecture diagram",
                    "mandatory": True,
                    "requiresUrl": True,
                },
            ],
        }
        raw = {
            "score": 98,
            "recommendation": "approve",
            "checks": [
                {
                    "key": "system_context",
                    "status": "met",
                    "severity": "info",
                    "evidence": "Users and boundaries",
                    "feedback": "Complete",
                }
            ],
        }

        result = _normalize_evaluation(request, raw)

        self.assertFalse(result.passed)
        self.assertEqual(result.recommendation, "changes_requested")
        self.assertEqual(result.score, 69)
        self.assertEqual(result.checks[1].status, "missing")
        self.assertTrue(result.revisionItems)

    def test_uiux_requires_approved_architecture(self):
        with self.assertRaisesRegex(ValueError, "approved architecture"):
            _validate_request_contract(
                {
                    "submission": {"submissionType": "ui_ux"},
                    "requirements": [{"key": "wireframes"}],
                }
            )


if __name__ == "__main__":
    unittest.main()
