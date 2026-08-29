import unittest

from app.agents.freelancer_matching import MatchFreelancersRequest, match_freelancers


class FreelancerMatchingTests(unittest.TestCase):
    def test_required_skills_and_availability_affect_rank(self):
        request = MatchFreelancersRequest(
            matchingRunId="run-1",
            targetType="task",
            targetRoleKey="backend",
            task={"title": "API", "requiredSkills": ["NestJS"]},
            candidates=[
                {
                    "freelancerProfileId": "strong",
                    "skills": ["NestJS"],
                    "availabilityHours": 20,
                    "yearsExperience": 4,
                    "hourlyRate": 20,
                },
                {
                    "freelancerProfileId": "weak",
                    "skills": ["Figma"],
                    "availabilityHours": 0,
                    "yearsExperience": 1,
                    "hourlyRate": 30,
                },
            ],
        )

        result = match_freelancers(request)

        self.assertEqual(result["candidates"][0]["freelancerProfileId"], "strong")
        self.assertEqual(result["candidates"][0]["rank"], 1)
        self.assertIn("no_availability", result["candidates"][1]["evidence"]["riskFlags"])

    def test_assessed_professional_role_is_a_soft_ranking_signal(self):
        request = MatchFreelancersRequest(
            matchingRunId="run-role",
            targetType="task",
            targetRoleKey="backend",
            task={"title": "API", "requiredSkills": ["TypeScript"]},
            candidates=[
                {
                    "freelancerProfileId": "frontend",
                    "professionalRole": "frontend",
                    "skills": ["TypeScript"],
                    "availabilityHours": 20,
                },
                {
                    "freelancerProfileId": "backend",
                    "professionalRole": "backend",
                    "skills": ["TypeScript"],
                    "availabilityHours": 20,
                },
            ],
        )

        result = match_freelancers(request)

        self.assertEqual(result["candidates"][0]["freelancerProfileId"], "backend")
        self.assertEqual(len(result["candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
