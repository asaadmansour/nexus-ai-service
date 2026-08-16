import unittest

from app.agents.submission_evaluation import (
    DEFAULT_IMPLEMENTATION_QUALITY_CRITERIA,
    RubricItem,
    SUBMISSION_EVALUATION_SYSTEM_PROMPT,
    SubmissionEvaluationResponse,
    _build_prompt,
    _normalize,
    _required_rubric_criteria,
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
            "evaluationHistory": [
                {
                    "commitSha": "a" * 40,
                    "recommendation": "changes_requested",
                    "unmetCriteria": ["Feature works"],
                }
            ],
        }

        result = _normalize(response, request)

        self.assertFalse(result["passed"])
        self.assertTrue(result["revisionRequested"])
        self.assertLessEqual(result["score"], 69)
        self.assertEqual(
            len(result["rubric"]),
            2 + len(DEFAULT_IMPLEMENTATION_QUALITY_CRITERIA),
        )
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

    def test_code_submission_enforces_requirements_and_engineering_quality(self):
        request = {
            "task": {
                "acceptanceCriteria": ["Endpoint returns the documented response"],
                "deliverables": ["Production-ready endpoint"],
                "integrationChecks": ["Consumer contract tests pass"],
                "contractReferences": ["API contract: POST /orders"],
                "ownedPaths": ["src/orders/**", "tests/orders/**"],
            },
            "submission": {"submissionType": "pull_request"},
        }

        criteria = _required_rubric_criteria(request)

        self.assertEqual(criteria[:3], [
            "Endpoint returns the documented response",
            "Production-ready endpoint",
            "Consumer contract tests pass",
        ])
        for quality_criterion in DEFAULT_IMPLEMENTATION_QUALITY_CRITERIA:
            self.assertIn(quality_criterion, criteria)
        self.assertIn(
            "Changes respect the assigned owned paths unless a documented "
            "integration exception is necessary: src/orders/**, tests/orders/**",
            criteria,
        )

    def test_omitted_quality_findings_fail_closed(self):
        response = SubmissionEvaluationResponse(
            passed=True,
            score=100,
            revisionRequested=False,
            revisionNotes="",
            requiresHumanReview=False,
            rubric=[
                RubricItem(
                    criterion="Feature works",
                    met=True,
                    evidence="A supplied test report demonstrates the behavior.",
                )
            ],
        )
        request = {
            "task": {"acceptanceCriteria": ["Feature works"]},
            "submission": {"submissionType": "repo"},
        }

        result = _normalize(response, request)

        self.assertFalse(result["passed"])
        self.assertTrue(result["revisionRequested"])
        self.assertTrue(result["requiresHumanReview"])
        self.assertLessEqual(result["score"], 69)
        self.assertTrue(
            any(
                item["criterion"] == DEFAULT_IMPLEMENTATION_QUALITY_CRITERIA[1]
                and item["met"] is False
                for item in result["rubric"]
            )
        )

    def test_prompt_contains_explicit_evidence_and_quality_policy(self):
        request = {
            "task": {
                "acceptanceCriteria": ["Feature works"],
                "qualityCriteria": DEFAULT_IMPLEMENTATION_QUALITY_CRITERIA,
            },
            "submission": {"submissionType": "pull_request"},
        }

        prompt = _build_prompt(request)

        self.assertIn("Required rubric criteria", prompt)
        self.assertIn(DEFAULT_IMPLEMENTATION_QUALITY_CRITERIA[0], prompt)
        self.assertIn("SOLID", SUBMISSION_EVALUATION_SYSTEM_PROMPT)
        self.assertIn("A URL, commit SHA", SUBMISSION_EVALUATION_SYSTEM_PROMPT)
        self.assertIn("evaluationHistory", prompt)
        self.assertIn("consistency ledger", SUBMISSION_EVALUATION_SYSTEM_PROMPT)

    def test_complete_sandbox_inspection_does_not_force_manual_review(self):
        request = {
            "task": {"acceptanceCriteria": ["Feature works"]},
            "submission": {
                "submissionType": "pull_request",
                "inspection": {
                    "complete": True,
                    "sourceInspected": True,
                    "snapshotVerified": True,
                    "verificationComplete": True,
                    "commitSha": "a" * 40,
                    "diffTruncated": False,
                    "coverage": {"changedFileCoverage": 1.0},
                    "verification": {"coverage": {"test": True}, "results": []},
                    "githubChecks": {"checkRuns": []},
                },
            },
        }
        criteria = _required_rubric_criteria(request)
        response = SubmissionEvaluationResponse(
            passed=True,
            score=90,
            revisionRequested=False,
            revisionNotes="",
            requiresHumanReview=False,
            rubric=[
                RubricItem(criterion=criterion, met=True, evidence="Inspected evidence")
                for criterion in criteria
            ],
        )

        result = _normalize(response, request)

        self.assertTrue(result["passed"])
        self.assertFalse(result["requiresHumanReview"])

    def test_failed_verification_overrides_model_pass_claim(self):
        request = {
            "task": {"acceptanceCriteria": ["Feature works"]},
            "submission": {
                "submissionType": "pull_request",
                "inspection": {
                    "complete": True,
                    "sourceInspected": True,
                    "snapshotVerified": True,
                    "verificationComplete": True,
                    "commitSha": "a" * 40,
                    "diffTruncated": False,
                    "coverage": {"changedFileCoverage": 1.0},
                    "verification": {
                        "coverage": {"test": True},
                        "results": [
                            {
                                "project": ".",
                                "category": "test",
                                "name": "node-tests",
                                "status": "failed",
                                "exitCode": 1,
                                "output": "One regression failed",
                            }
                        ]
                    },
                    "githubChecks": {"checkRuns": []},
                },
            },
        }
        criteria = _required_rubric_criteria(request)
        response = SubmissionEvaluationResponse(
            passed=True,
            score=99,
            revisionRequested=False,
            revisionNotes="",
            requiresHumanReview=False,
            rubric=[
                RubricItem(criterion=criterion, met=True, evidence="Model claimed pass")
                for criterion in criteria
            ],
        )

        result = _normalize(response, request)

        failed = next(
            item for item in result["rubric"]
            if item["criterion"] == "Automated test check passes: . — node-tests"
        )
        self.assertFalse(failed["met"])
        self.assertIn("One regression failed", failed["evidence"])
        self.assertFalse(result["passed"])
        self.assertTrue(result["revisionRequested"])

    def test_pending_github_check_requires_review_without_requesting_revision(self):
        request = {
            "task": {"acceptanceCriteria": ["Feature works"]},
            "submission": {
                "submissionType": "pull_request",
                "inspection": {
                    "complete": True,
                    "sourceInspected": True,
                    "snapshotVerified": True,
                    "verificationComplete": True,
                    "commitSha": "a" * 40,
                    "diffTruncated": False,
                    "coverage": {"changedFileCoverage": 1.0},
                    "verification": {
                        "coverage": {"test": True},
                        "results": [],
                    },
                    "githubChecks": {
                        "checkRuns": [
                            {
                                "name": "CI",
                                "status": "in_progress",
                                "conclusion": None,
                            }
                        ]
                    },
                },
            },
        }
        criteria = _required_rubric_criteria(request)
        response = SubmissionEvaluationResponse(
            passed=True,
            score=90,
            revisionRequested=False,
            revisionNotes="",
            requiresHumanReview=False,
            rubric=[
                RubricItem(criterion=criterion, met=True, evidence="Verified")
                for criterion in criteria
            ],
        )

        result = _normalize(response, request)

        self.assertTrue(result["passed"])
        self.assertTrue(result["requiresHumanReview"])
        self.assertFalse(result["revisionRequested"])


if __name__ == "__main__":
    unittest.main()
