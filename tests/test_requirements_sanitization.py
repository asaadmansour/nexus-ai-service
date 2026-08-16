import unittest

from app.agents.requirements.llm import _normalize_llm_result
from app.agents.requirements.nodes import check_missing_fields_node


class RequirementsSanitizationTests(unittest.TestCase):
    def test_questions_and_placeholders_are_not_stored_as_requirements(self):
        result = _normalize_llm_result(
            {
                "extractedFields": {
                    "mainGoal": "what do you mean?",
                    "targetUsers": "not sure",
                    "coreFeatures": ["like what?", "Product catalog", "source code"],
                    "platforms": "which one?",
                    "deliverables": ["live link", "documentation"],
                },
                "assistantReply": "Here are some examples.",
            }
        )

        self.assertEqual(
            result["extractedFields"],
            {
                "coreFeatures": ["Product catalog"],
                "deliverables": ["live link", "documentation"],
            },
        )

    def test_deliverable_names_are_kept_out_of_core_features(self):
        result = _normalize_llm_result(
            {
                "extractedFields": {
                    "coreFeatures": [
                        "User login",
                        "live link",
                        "deployment help",
                    ]
                },
                "assistantReply": None,
            }
        )

        self.assertEqual(result["extractedFields"], {"coreFeatures": ["User login"]})

    def test_optional_staffing_and_preference_fields_do_not_block_a_brief(self):
        result = check_missing_fields_node(
            {
                "mergedBrief": {
                    "mainGoal": "Display Hello World",
                    "targetUsers": ["website visitors"],
                    "coreFeatures": ["Display Hello World"],
                    "platforms": ["website"],
                    "deliverables": ["live link"],
                }
            }
        )

        self.assertTrue(result["isComplete"])
        self.assertEqual(result["missingFields"], [])
        self.assertEqual(result["completionPercentage"], 100)


if __name__ == "__main__":
    unittest.main()
