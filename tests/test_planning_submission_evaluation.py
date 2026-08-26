import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import errors

from app.agents.planning_submission_evaluation import (
    DEFAULT_GEMINI_MODEL,
    ModelEvaluationResponse,
    PROMPT_VERSION,
    _context_hash,
    _evaluation_input_hash,
    _generate_evaluation_response,
    _get_model_candidates,
    _normalize_evaluation,
    _validate_request_contract,
    evaluate_submission,
)
from app.agents.planning_artifacts import (
    ArtifactInspectionError,
    MAX_FIGMA_STRUCTURE_FRAMES,
    _collect_urls,
    _figma_structure,
    _validate_public_https_url,
)
from app.runners.planning_evaluation import _summary


class PlanningSubmissionEvaluationTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_uses_supported_shared_default_when_environment_is_missing(self):
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-3.1-flash-lite")
        self.assertEqual(_get_model_candidates(), [DEFAULT_GEMINI_MODEL])

    @patch.dict(
        "os.environ",
        {"GEMINI_MODEL": "working-model", "GEMINI_FALLBACK_MODELS": ""},
        clear=False,
    )
    def test_provider_schema_rejection_retries_in_json_only_mode(self):
        calls = []

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise errors.APIError(
                        400,
                        {"error": {"message": "Schema is too complex"}},
                    )
                return SimpleNamespace(text="{}")

        response, model = _generate_evaluation_response(
            SimpleNamespace(models=Models()), ["prompt"]
        )

        self.assertEqual(model, "working-model")
        self.assertEqual(response.text, "{}")
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(calls[0]["config"].response_json_schema)
        self.assertIsNone(calls[1]["config"].response_json_schema)

    def test_gemini_response_schema_has_no_open_dictionary_fields(self):
        schema = ModelEvaluationResponse.model_json_schema()

        self.assertNotIn("additionalProperties", json.dumps(schema))

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
        # One of two applicable requirements is met. The score is derived from
        # criterion statuses, then capped when blockers exist; it must not reuse
        # the model's contradictory 98/100 claim.
        self.assertEqual(result.score, 50)
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

    def test_required_artifact_needs_a_real_citation(self):
        request = {
            "submission": {
                "submissionType": "architecture",
                "content": {
                    "requirementEvidence": {
                        "diagram": {
                            "summary": "Diagram supplied",
                            "urls": ["https://example.com/diagram.pdf"],
                        }
                    }
                },
            },
            "requirements": [
                {
                    "key": "diagram",
                    "title": "Diagram",
                    "mandatory": True,
                    "requiresUrl": True,
                }
            ],
        }
        manifest = {
            "manifestHash": "manifest",
            "artifacts": [
                {
                    "id": "artifact-1",
                    "status": "inspected",
                    "requirementKeys": ["diagram"],
                }
            ],
        }
        raw = {
            "score": 95,
            "recommendation": "approve",
            "checks": [
                {
                    "key": "diagram",
                    "status": "met",
                    "severity": "info",
                    "evidence": "Diagram supplied",
                    "feedback": "Complete",
                    "citations": [],
                }
            ],
        }

        result = _normalize_evaluation(request, raw, manifest=manifest)

        self.assertEqual(result.checks[0].status, "partial")
        self.assertEqual(result.checks[0].severity, "blocker")
        self.assertFalse(result.passed)

    def test_justified_not_applicable_is_satisfied_without_artifact(self):
        request = {
            "submission": {
                "submissionType": "architecture",
                "content": {
                    "requirementEvidence": {
                        "api_contract": {
                            "disposition": "not_applicable",
                            "notApplicableReason": (
                                "The approved static page has no runtime API or server."
                            ),
                            "summary": "",
                            "urls": [],
                        }
                    }
                },
            },
            "requirements": [
                {
                    "key": "api_contract",
                    "title": "API contract",
                    "mandatory": True,
                    "requiresUrl": True,
                    "allowNotApplicable": True,
                }
            ],
        }
        raw = {
            "score": 100,
            "recommendation": "approve",
            "checks": [
                {
                    "key": "api_contract",
                    "status": "not_applicable",
                    "severity": "info",
                    "evidence": "No runtime API or server.",
                    "feedback": "Consistent with the static architecture.",
                }
            ],
        }

        result = _normalize_evaluation(request, raw)

        self.assertTrue(result.passed)
        self.assertEqual(result.checks[0].status, "not_applicable")
        self.assertEqual(result.openIssues, [])

    def test_invalid_not_applicable_claim_is_a_blocking_conflict(self):
        request = {
            "submission": {
                "submissionType": "architecture",
                "content": {
                    "requirementEvidence": {
                        "system_context": {
                            "disposition": "not_applicable",
                            "notApplicableReason": "Not needed",
                        }
                    }
                },
            },
            "requirements": [
                {
                    "key": "system_context",
                    "title": "System context",
                    "mandatory": True,
                    "requiresUrl": False,
                    "allowNotApplicable": False,
                }
            ],
        }
        raw = {
            "score": 100,
            "recommendation": "approve",
            "checks": [
                {
                    "key": "system_context",
                    "status": "not_applicable",
                    "severity": "info",
                    "evidence": "Not needed",
                    "feedback": "N/A",
                }
            ],
        }

        result = _normalize_evaluation(request, raw)

        self.assertFalse(result.passed)
        self.assertEqual(result.checks[0].status, "conflict")
        self.assertEqual(result.checks[0].severity, "blocker")

    def test_omitted_optional_requirement_does_not_create_an_issue(self):
        request = {
            "submission": {
                "submissionType": "ui_ux",
                "content": {"requirementEvidence": {}},
            },
            "requirements": [
                {
                    "key": "prototype",
                    "title": "Prototype",
                    "mandatory": False,
                    "requiresUrl": False,
                    "allowNotApplicable": True,
                }
            ],
        }
        raw = {
            "score": 100,
            "recommendation": "approve",
            "checks": [],
        }

        result = _normalize_evaluation(request, raw)

        self.assertTrue(result.passed)
        self.assertEqual(result.checks[0].status, "not_applicable")
        self.assertEqual(result.openIssues, [])

    def test_not_applicable_evidence_urls_are_not_downloaded(self):
        urls = _collect_urls(
            {
                "submission": {
                    "content": {
                        "requirementEvidence": {
                            "api_contract": {
                                "disposition": "not_applicable",
                                "urls": ["https://example.com/legacy-api.pdf"],
                            },
                            "screen_designs": {
                                "disposition": "covered",
                                "urls": ["https://example.com/screen.png"],
                            },
                        }
                    }
                }
            }
        )

        self.assertEqual(
            urls,
            [("https://example.com/screen.png", ["screen_designs"])],
        )

    def test_identical_snapshot_reuses_previous_verdict(self):
        request = {
            "project": {"id": "project"},
            "brief": {},
            "submission": {
                "submissionId": "new-version",
                "submissionVersion": 2,
                "submissionType": "architecture",
                "content": {
                    "requirementEvidence": {
                        "context": {"summary": "Defined", "urls": []}
                    }
                },
                "fileUrls": {},
            },
            "requirements": [
                {
                    "key": "context",
                    "title": "Context",
                    "mandatory": True,
                    "requiresUrl": False,
                }
            ],
        }
        previous_request = deepcopy(request)
        previous_request["submission"]["submissionId"] = "old-version"
        previous_request["submission"]["submissionVersion"] = 1
        manifest = {"manifestHash": "same", "artifacts": [], "totalBytes": 0}
        context_hash = _context_hash(previous_request)
        self.assertEqual(context_hash, _context_hash(request))
        input_hash = _evaluation_input_hash(context_hash, manifest)
        previous = _normalize_evaluation(
            previous_request,
            {
                "score": 90,
                "recommendation": "approve",
                "checks": [
                    {
                        "key": "context",
                        "status": "met",
                        "severity": "info",
                        "evidence": "Defined",
                        "feedback": "Complete",
                    }
                ],
            },
            manifest=manifest,
            evaluation_input_hash=input_hash,
            context_hash=context_hash,
            model_name="model",
        ).model_dump()
        previous["promptVersion"] = PROMPT_VERSION
        request["previousVerdict"] = previous

        with patch(
            "app.agents.planning_submission_evaluation.inspect_artifacts",
            return_value=(manifest, [], ""),
        ):
            result = evaluate_submission(request)

        self.assertTrue(result["reused"])
        self.assertEqual(result["evaluationInputHash"], input_hash)

    def test_citation_must_belong_to_the_requirement(self):
        request = {
            "submission": {
                "submissionType": "architecture",
                "content": {
                    "requirementEvidence": {
                        "diagram": {
                            "summary": "Diagram supplied",
                            "urls": ["https://example.com/diagram.pdf"],
                        }
                    }
                },
            },
            "requirements": [
                {
                    "key": "diagram",
                    "title": "Diagram",
                    "mandatory": True,
                    "requiresUrl": True,
                }
            ],
        }
        manifest = {
            "manifestHash": "manifest",
            "artifacts": [
                {
                    "id": "artifact-diagram",
                    "status": "inspected",
                    "requirementKeys": ["diagram"],
                },
                {
                    "id": "artifact-unrelated",
                    "status": "inspected",
                    "requirementKeys": ["data_model"],
                },
            ],
        }
        raw = {
            "score": 95,
            "recommendation": "approve",
            "checks": [
                {
                    "key": "diagram",
                    "status": "met",
                    "severity": "info",
                    "evidence": "Diagram supplied",
                    "feedback": "Complete",
                    "citations": [
                        {
                            "artifactId": "artifact-unrelated",
                            "location": "page 1",
                            "finding": "Unrelated evidence",
                        }
                    ],
                }
            ],
        }

        result = _normalize_evaluation(request, raw, manifest=manifest)

        self.assertEqual(result.checks[0].status, "partial")
        self.assertEqual(result.checks[0].citations, [])
        self.assertFalse(result.passed)

    def test_private_network_artifact_is_rejected(self):
        with self.assertRaises(ArtifactInspectionError):
            _validate_public_https_url("https://127.0.0.1/private.pdf")

    def test_large_figma_structure_is_bounded(self):
        frames = [
            {"id": str(index), "name": f"Frame {index}", "type": "FRAME"}
            for index in range(MAX_FIGMA_STRUCTURE_FRAMES + 1)
        ]
        structure = _figma_structure(
            {
                "document": {
                    "id": "root",
                    "name": "Document",
                    "type": "DOCUMENT",
                    "children": [
                        {
                            "id": "page",
                            "name": "Page",
                            "type": "CANVAS",
                            "children": frames,
                        }
                    ],
                }
            }
        )

        self.assertEqual(len(structure["frames"]), MAX_FIGMA_STRUCTURE_FRAMES)
        self.assertEqual(structure["frameCount"], MAX_FIGMA_STRUCTURE_FRAMES + 1)
        self.assertTrue(structure["structureTruncated"])

    def test_sandbox_summary_keeps_prior_and_current_verdicts(self):
        summary = _summary(
            {
                "recommendation": "changes_requested",
                "score": 60,
                "evaluationInputHash": "old-hash",
                "summary": "Prior review",
                "openIssues": [
                    {
                        "criterionKey": "api_contracts",
                        "message": "Document response schemas",
                    }
                ],
            },
            {
                "recommendation": "approve",
                "score": 91,
                "evaluationInputHash": "new-hash",
                "summary": "Current review",
                "openIssues": [],
                "resolvedIssues": ["planning-issue"],
                "regressions": [],
                "reused": False,
            },
        )

        self.assertIn("## Prior verdict", summary)
        self.assertIn("Document response schemas", summary)
        self.assertIn("## Current verdict", summary)
        self.assertIn("planning-issue", summary)


if __name__ == "__main__":
    unittest.main()
