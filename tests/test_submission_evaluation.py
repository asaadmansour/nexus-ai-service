import unittest
from unittest.mock import patch

from app.agents.submission_evaluation import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_IMPLEMENTATION_QUALITY_CRITERIA,
    RubricItem,
    SUBMISSION_EVALUATION_SYSTEM_PROMPT,
    SubmissionEvaluationResponse,
    _build_prompt,
    _get_model_candidates,
    _normalize,
    _required_rubric_criteria,
)
from app.routers.evaluation_agent import EvaluateSubmissionRequest


class SubmissionEvaluationTests(unittest.TestCase):
    def test_request_normalizes_nullable_legacy_list_fields(self):
        request = EvaluateSubmissionRequest.model_validate(
            {
                "project": {"projectId": "project-1"},
                "task": {
                    "taskId": "task-1",
                    "title": "Final delivery",
                    "acceptanceCriteria": None,
                    "deliverables": None,
                    "qualityCriteria": None,
                },
                "submission": {
                    "submissionId": "submission-1",
                    "submissionType": "repo",
                },
                "brief": {"acceptanceCriteria": None},
                "evaluationHistory": None,
            }
        )

        self.assertEqual(request.task.acceptance_criteria, [])
        self.assertEqual(request.task.deliverables, [])
        self.assertEqual(request.task.quality_criteria, [])
        self.assertEqual(request.brief.acceptance_criteria, [])
        self.assertEqual(request.evaluation_history, [])

    @patch.dict("os.environ", {}, clear=True)
    def test_uses_supported_shared_default_when_environment_is_missing(self):
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-3.1-flash-lite")
        self.assertEqual(_get_model_candidates(), [DEFAULT_GEMINI_MODEL])

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
                    status="met",
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
            rubric=[
                RubricItem(
                    criterion="Explain the algorithm",
                    status="met",
                    met=True,
                    evidence="Done",
                )
            ],
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
        for quality_criterion in DEFAULT_IMPLEMENTATION_QUALITY_CRITERIA[:3]:
            self.assertIn(quality_criterion, criteria)
        self.assertIn(
            "Automated tests cover the changed behavior and relevant failure or regression paths, and the supplied test evidence passes.",
            criteria,
        )
        self.assertNotIn(DEFAULT_IMPLEMENTATION_QUALITY_CRITERIA[3], criteria)
        self.assertIn(
            "Changes respect the assigned owned paths unless a documented "
            "integration exception is necessary: src/orders/**, tests/orders/**",
            criteria,
        )

    def test_static_console_smoke_check_does_not_require_automated_tests(self):
        request = {
            "task": {
                "title": "Hello World page",
                "acceptanceCriteria": ["The page displays Hello World"],
                "integrationChecks": ["The browser console has no errors"],
                "deliverables": ["Live deployment URL"],
            },
            "submission": {"submissionType": "pull_request"},
        }

        criteria = _required_rubric_criteria(request)

        self.assertIn(DEFAULT_IMPLEMENTATION_QUALITY_CRITERIA[3], criteria)
        self.assertFalse(
            any(criterion.startswith("Automated tests cover") for criterion in criteria)
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
                    status="met",
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

        self.assertIn("Required rubric definitions", prompt)
        self.assertIn(DEFAULT_IMPLEMENTATION_QUALITY_CRITERIA[0], prompt)
        self.assertIn("SOLID", SUBMISSION_EVALUATION_SYSTEM_PROMPT)
        self.assertIn("A URL, commit SHA", SUBMISSION_EVALUATION_SYSTEM_PROMPT)
        self.assertIn("evaluationHistory", prompt)
        self.assertIn("consistency ledger", SUBMISSION_EVALUATION_SYSTEM_PROMPT)

    def test_justified_not_applicable_is_satisfied_only_when_allowed(self):
        criterion = {
            "key": "security_auth_privacy",
            "criterion": "Authentication and privacy controls touched by the task are correct.",
            "category": "security",
            "mandatory": True,
            "allowNotApplicable": True,
            "rationale": "Potentially relevant security concern.",
        }
        request = {
            "task": {"evaluationCriteria": [criterion]},
            "submission": {
                "submissionType": "text",
                "submissionText": "Static copy-only change",
            },
        }
        response = SubmissionEvaluationResponse(
            passed=True,
            score=90,
            revisionRequested=False,
            revisionNotes="",
            requiresHumanReview=False,
            rubric=[
                RubricItem(
                    key=criterion["key"],
                    criterion=criterion["criterion"],
                    category="security",
                    status="not_applicable",
                    met=True,
                    evidence="The inspected static submission contains no authentication or personal-data behavior.",
                )
            ],
        )

        result = _normalize(response, request)

        self.assertTrue(result["passed"])
        self.assertEqual(result["rubric"][0]["status"], "not_applicable")
        self.assertTrue(result["rubric"][0]["met"])

        request["task"]["evaluationCriteria"][0]["allowNotApplicable"] = False
        rejected = _normalize(response, request)
        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["rubric"][0]["status"], "unmet")

    def test_missing_executable_test_evidence_routes_to_manual_review(self):
        test_criterion = {
            "key": "verification_automated_tests",
            "criterion": "Automated tests cover the changed behavior.",
            "category": "verification",
            "mandatory": True,
            "allowNotApplicable": False,
            "rationale": "This API change needs regression protection.",
        }
        request = {
            "task": {"evaluationCriteria": [test_criterion]},
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
                    "verification": {"coverage": {"test": False}, "results": []},
                    "githubChecks": {"checkRuns": []},
                },
            },
        }
        criteria = _required_rubric_criteria(request)
        response = SubmissionEvaluationResponse(
            passed=True,
            score=95,
            revisionRequested=False,
            revisionNotes="",
            requiresHumanReview=False,
            rubric=[
                RubricItem(
                    criterion=item,
                    status="met",
                    met=True,
                    evidence="Claimed as complete",
                )
                for item in criteria
            ],
        )

        result = _normalize(response, request)

        test_row = next(
            item for item in result["rubric"] if item["key"] == "verification_automated_tests"
        )
        self.assertFalse(test_row["met"])
        self.assertEqual(test_row["status"], "unverified")
        self.assertIn("No executable automated test", test_row["evidence"])
        self.assertTrue(result["passed"])
        self.assertTrue(result["requiresHumanReview"])
        self.assertFalse(result["revisionRequested"])

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
                RubricItem(
                    criterion=criterion,
                    status="met",
                    met=True,
                    evidence="Inspected evidence",
                )
                for criterion in criteria
            ],
        )

        result = _normalize(response, request)

        self.assertTrue(result["passed"])
        self.assertFalse(result["requiresHumanReview"])

    def test_incomplete_inspection_routes_to_human_without_revision(self):
        request = {
            "task": {"acceptanceCriteria": ["Feature works"]},
            "submission": {
                "submissionType": "pull_request",
                "inspection": {
                    "complete": False,
                    "sourceInspected": True,
                    "snapshotVerified": True,
                    "verificationComplete": True,
                    "commitSha": "a" * 40,
                    "diffTruncated": True,
                    "changedFilesTruncated": False,
                    "coverage": {"changedFileCoverage": 0.5},
                    "verification": {"coverage": {"test": True}, "results": []},
                    "githubChecks": {"checkRuns": []},
                    "pullRequest": {
                        "number": 4,
                        "state": "open",
                        "draft": False,
                        "headSha": "a" * 40,
                    },
                },
            },
        }
        criteria = _required_rubric_criteria(request)
        response = SubmissionEvaluationResponse(
            passed=False,
            score=91,
            revisionRequested=True,
            revisionNotes="Source coverage was incomplete.",
            requiresHumanReview=True,
            rubric=[
                RubricItem(
                    criterion=criterion,
                    status="met",
                    met=True,
                    evidence="The available evidence supports this row.",
                )
                for criterion in criteria
            ],
        )

        result = _normalize(response, request)

        visibility = next(
            item
            for item in result["rubric"]
            if item["key"] == "verification_observed_1"
        )
        self.assertEqual(visibility["status"], "unverified")
        self.assertFalse(visibility["met"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["score"], 91)
        self.assertTrue(result["requiresHumanReview"])
        self.assertFalse(result["revisionRequested"])

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
                RubricItem(
                    criterion=criterion,
                    status="met",
                    met=True,
                    evidence="Model claimed pass",
                )
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
                RubricItem(
                    criterion=criterion,
                    status="met",
                    met=True,
                    evidence="Verified",
                )
                for criterion in criteria
            ],
        )

        result = _normalize(response, request)

        self.assertTrue(result["passed"])
        self.assertTrue(result["requiresHumanReview"])
        self.assertFalse(result["revisionRequested"])


if __name__ == "__main__":
    unittest.main()
