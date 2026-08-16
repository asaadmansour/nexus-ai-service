import unittest

from app.agents.project_plan_generation import (
    ProjectPlanGenerationError,
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
            "architecture": {"services": ["api"]},
            "designSystem": {"tokens": ["primary"]},
            "apiContract": {"endpoints": ["GET /api/example"]},
            "dataModel": {"entities": ["Example"]},
            "conventions": {"branching": "feature branches"},
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

        self.assertFalse(result.projectSpec.apiContract["applicable"])

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


if __name__ == "__main__":
    unittest.main()
