import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import errors

from app.agents.project_plan_generation import (
    ProjectPlanRequest,
    ProjectPlanGenerationError,
    _generate_plan_response,
    _get_model_candidates,
    generate_project_plan,
    validate_and_normalize_plan,
)
from app.routers.generate_project_plan import GeneratePlanRequest


def valid_plan():
    return {
        "summary": "Build from approved contracts.",
        "assumptions": [],
        "timeline": {"estimatedWeeks": 2, "phases": []},
        "milestones": [
            {
                "clientKey": "m1",
                "title": "Foundation",
                "description": "Core implementation",
                "orderIndex": 1,
                "startDay": 0,
                "estimatedDays": 10,
                "budgetAmount": 1000,
                "currency": "EGP",
                "acceptanceCriteria": ["Integrated build passes"],
            }
        ],
        "tasks": [
            {
                "clientKey": "t1",
                "milestoneClientKey": "m1",
                "title": "Contract",
                "description": "Implement contract",
                "priority": "high",
                "roleKey": "backend",
                "requiredSkills": ["NestJS"],
                "estimatedHours": 8,
                "orderIndex": 1,
                "startDay": 0,
                "durationDays": 2,
                "acceptanceCriteria": ["Contract tests pass"],
                "contractReferences": ["Architecture API contract"],
                "ownedPaths": ["src/contracts"],
                "integrationChecks": ["npm test"],
            },
            {
                "clientKey": "t2",
                "milestoneClientKey": "m1",
                "title": "Consumer",
                "description": "Use contract",
                "priority": "medium",
                "roleKey": "frontend",
                "requiredSkills": ["Next.js"],
                "estimatedHours": 8,
                "orderIndex": 2,
                "startDay": 2,
                "durationDays": 2,
                "acceptanceCriteria": ["Consumer test passes"],
                "contractReferences": ["UI screen-to-API map"],
                "ownedPaths": ["src/app"],
                "integrationChecks": ["npm run build"],
            },
        ],
        "dependencies": [
            {
                "taskClientKey": "t2",
                "dependsOnTaskClientKey": "t1",
                "dependencyType": "blocks",
            }
        ],
        "teamPlan": {
            "recommendedRoles": [
                {"roleKey": "backend", "count": 1, "skills": ["NestJS"]},
                {"roleKey": "frontend", "count": 1, "skills": ["Next.js"]},
            ],
            "suggestedTeamSize": 2,
        },
        "riskRegister": [],
        "projectSpec": {
            "architecture": {
                "summary": "Single API service",
                "decisions": ["Use the approved API module boundaries"],
            },
            "designSystem": {
                "summary": "Approved visual tokens",
                "decisions": ["Use the primary token set"],
            },
            "apiContract": {
                "summary": "Approved HTTP contract",
                "decisions": ["GET /api/example"],
            },
            "dataModel": {
                "summary": "Approved Example entity",
                "decisions": ["Persist Example records"],
            },
            "conventions": {
                "summary": "Repository conventions",
                "decisions": ["Use feature branches"],
            },
        },
    }


class ProjectPlanGenerationTests(unittest.TestCase):
    def test_valid_parallel_contract_plan_is_accepted(self):
        result = validate_and_normalize_plan(valid_plan())
        self.assertEqual(len(result.tasks), 2)

    def test_blocking_dependency_must_finish_before_dependent_task(self):
        plan = valid_plan()
        plan["tasks"][1]["startDay"] = 1
        with self.assertRaisesRegex(ProjectPlanGenerationError, "starts before"):
            validate_and_normalize_plan(plan)

    def test_explicit_not_applicable_spec_sections_are_accepted(self):
        plan = valid_plan()
        plan["projectSpec"]["apiContract"] = {
            "applicable": False,
            "reason": "The approved page is static and has no runtime API.",
        }
        plan["projectSpec"]["dataModel"] = {
            "applicable": False,
            "reason": "The approved page stores no persistent user or product data.",
        }

        result = validate_and_normalize_plan(plan)

        self.assertFalse(result.projectSpec.apiContract.applicable)

    def test_not_applicable_spec_sections_need_a_real_reason(self):
        plan = valid_plan()
        plan["projectSpec"]["apiContract"] = {
            "applicable": False,
            "reason": "No API",
        }

        with self.assertRaisesRegex(ProjectPlanGenerationError, "needs a concrete reason"):
            validate_and_normalize_plan(plan)

    def test_nullable_backend_descriptions_and_summaries_match_contract(self):
        request = GeneratePlanRequest(
            projectPlanJobId="job",
            project={
                "id": "project",
                "title": "Project",
                "description": None,
                "status": "planning_review",
                "budgetMin": 100,
                "budgetMax": 200,
                "currency": "EGP",
            },
            brief={},
            architectureSubmission={"id": "a", "summary": None, "content": {}},
            uiuxSubmission={"id": "u", "summary": None, "content": {}},
            planningTeam=[],
        )
        self.assertIsNone(request.architectureSubmission.summary)

    def test_backend_planning_context_is_not_dropped_by_route_schema(self):
        request = GeneratePlanRequest(
            projectPlanJobId="job",
            project={
                "id": "project",
                "title": "Project",
                "status": "planning_review",
                "budgetMin": 100,
                "budgetMax": 200,
                "currency": "EGP",
            },
            brief={"requirementProfile": {"complexity": "trivial"}},
            architectureSubmission={
                "id": "a",
                "content": {},
                "evaluationRequirements": {"profile": "trivial"},
                "evaluationResult": {"recommendation": "approve"},
                "adminNotes": "Approved architecture contract.",
            },
            uiuxSubmission={
                "id": "u",
                "content": {},
                "evaluationRequirements": {"profile": "trivial"},
            },
            planningTeam=[],
            notes="Keep the approved scope small.",
        )

        self.assertEqual(request.brief.requirementProfile["complexity"], "trivial")
        self.assertEqual(
            request.architectureSubmission.evaluationRequirements["profile"],
            "trivial",
        )
        self.assertEqual(
            request.architectureSubmission.evaluationResult["recommendation"],
            "approve",
        )
        self.assertEqual(
            request.architectureSubmission.adminNotes,
            "Approved architecture contract.",
        )
        self.assertEqual(request.notes, "Keep the approved scope small.")

    @patch.dict(
        "os.environ",
        {
            "GEMINI_PLAN_MODEL": "unavailable-model",
            "GEMINI_PLAN_FALLBACK_MODELS": "working-model",
            "GEMINI_PLAN_MAX_OUTPUT_TOKENS": "24000",
            "GEMINI_PLAN_TIMEOUT_MS": "180000",
        },
        clear=False,
    )
    def test_plan_generation_falls_back_and_uses_structured_output(self):
        calls = []

        class Models:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                if kwargs["model"] == "unavailable-model":
                    raise errors.APIError(404, {"error": "model unavailable"})
                return SimpleNamespace(parsed=valid_plan(), text=None)

        response, model = _generate_plan_response(
            SimpleNamespace(models=Models()), "prompt"
        )

        self.assertEqual(model, "working-model")
        self.assertEqual(response.parsed["summary"], "Build from approved contracts.")
        self.assertEqual([call["model"] for call in calls], [
            "unavailable-model",
            "working-model",
        ])
        config = calls[-1]["config"]
        self.assertEqual(config.max_output_tokens, 24000)
        self.assertEqual(config.http_options.timeout, 180000)
        self.assertIsNotNone(config.response_schema)
        self.assertIn("untrusted project data", config.system_instruction)

    @patch("app.agents.project_plan_generation.genai.Client")
    def test_generation_accepts_sdk_parsed_response(self, client_factory):
        client_factory.return_value.models.generate_content.return_value = (
            SimpleNamespace(parsed=valid_plan(), text=None)
        )
        request = ProjectPlanRequest(
            projectPlanJobId="job",
            project={"id": "project"},
            brief={"requirementProfile": {"complexity": "standard"}},
            architectureSubmission={"id": "a", "content": {}},
            uiuxSubmission={"id": "u", "content": {}},
            planningTeam=[],
            notes="Use approved contracts.",
        )

        result = generate_project_plan(request)

        self.assertEqual(result["summary"], "Build from approved contracts.")
        prompt = client_factory.return_value.models.generate_content.call_args.kwargs[
            "contents"
        ][0]
        self.assertIn("Use approved contracts.", prompt)

    @patch.dict(
        "os.environ",
        {"GEMINI_PLAN_MODEL": "custom-plan", "GEMINI_PLAN_FALLBACK_MODELS": ""},
        clear=False,
    )
    def test_stable_model_is_always_available_as_a_fallback(self):
        models = _get_model_candidates()
        self.assertEqual(models[0], "custom-plan")
        self.assertIn("gemini-2.5-flash", models)

    def test_provider_schema_contains_no_unsupported_open_dictionaries(self):
        from app.agents.project_plan_generation import ProjectPlanResponse

        def walk(value):
            if isinstance(value, dict):
                self.assertNotIn("additionalProperties", value)
                for nested in value.values():
                    walk(nested)
            elif isinstance(value, list):
                for nested in value:
                    walk(nested)

        walk(ProjectPlanResponse.model_json_schema())


if __name__ == "__main__":
    unittest.main()
