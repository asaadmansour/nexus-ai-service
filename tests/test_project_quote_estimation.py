import unittest
from unittest.mock import patch

from app.agents.project_quote_estimation import (
    ProjectQuoteRequest,
    _build_prompt,
    _fallback_quote,
    _is_minimal_website_scope,
    estimate_project_quote,
)


class ProjectQuoteEstimationTests(unittest.TestCase):
    def request(self, budget_max: float) -> ProjectQuoteRequest:
        return ProjectQuoteRequest(
            project={
                "budgetMin": 500,
                "budgetMax": budget_max,
                "currency": "egp",
            },
            brief={
                "coreFeatures": ["Render one Hello World page"],
                "platforms": ["web"],
                "deliverables": ["source code"],
                "suggestedTeamSize": 1,
                "requirementProfile": {"complexity": "trivial"},
            },
        )

    @patch.dict(
        "os.environ",
        {
            "MARKET_RATE_PRINCIPAL_REVIEWER": "650",
            "MARKET_RATE_ARCHITECT": "550",
            "MARKET_RATE_UI_UX": "450",
            "MARKET_RATE_DEVELOPER": "400",
        },
        clear=False,
    )
    def test_market_cost_is_not_clamped_to_an_insufficient_customer_budget(self):
        quote = _fallback_quote(self.request(1000), "fallback")

        self.assertEqual(quote["quoteStatus"], "out_of_budget")
        self.assertGreater(quote["recommendedMinimum"], 1000)
        self.assertEqual(
            quote["budgetGap"],
            round(quote["recommendedMinimum"] - 1000, 2),
        )
        self.assertEqual(
            {row["roleKey"] for row in quote["roleEstimates"]},
            {"principal_reviewer", "architect", "ui_ux", "implementation"},
        )

    def test_sufficient_budget_remains_payable(self):
        quote = _fallback_quote(self.request(100_000), "fallback")

        self.assertEqual(quote["quoteStatus"], "pending_customer")
        self.assertEqual(quote["budgetGap"], 0)
        self.assertEqual(quote["amount"], quote["recommendedMinimum"])

    def test_mobile_friendly_landing_page_uses_minimal_scope(self):
        brief = {
            "solutionType": "mobile-friendly landing page",
            "platforms": ["website"],
            "scopeDetails": "one page with five static sections",
            "integrations": "none",
            "adminNeeds": "no admin dashboard",
            "coreFeatures": ["marketing content", "contact details"],
            "deliverables": ["source code", "live link"],
        }

        self.assertTrue(_is_minimal_website_scope(brief))
        quote = _fallback_quote(
            ProjectQuoteRequest(
                project={"budgetMin": 500, "budgetMax": 100_000, "currency": "EGP"},
                brief=brief,
            ),
            "fallback",
        )

        implementation = next(
            row for row in quote["roleEstimates"] if row["roleKey"] == "implementation"
        )
        self.assertEqual(implementation["hoursEach"], 8)
        self.assertEqual(quote["complexity"], "low")

    def test_native_mobile_app_is_not_minimal_website_scope(self):
        self.assertFalse(
            _is_minimal_website_scope(
                {
                    "solutionType": "landing page and Android mobile app",
                    "platforms": ["website", "mobile app"],
                    "integrations": "none",
                    "adminNeeds": "none",
                }
            )
        )

    @patch.dict(
        "os.environ",
        {
            "MARKET_RATE_PRINCIPAL_REVIEWER": "650",
            "MARKET_RATE_ARCHITECT": "550",
            "MARKET_RATE_UI_UX": "450",
            "MARKET_RATE_DEVELOPER": "400",
        },
        clear=False,
    )
    def test_customer_budget_minimum_does_not_inflate_the_quote(self):
        request = ProjectQuoteRequest(
            project={"budgetMin": 275_000, "budgetMax": 300_000, "currency": "EGP"},
            brief={
                "mainGoal": "Present the business and collect customer enquiries",
                "targetUsers": ["potential customers"],
                "coreFeatures": ["Show service information", "Contact enquiry form"],
                "platforms": ["website"],
                "solutionType": "single landing page",
                "scopeDetails": "one page with five static sections",
                "integrations": "none",
                "adminNeeds": "no admin dashboard",
                "deliverables": ["working website", "source code", "live link"],
            },
        )

        quote = estimate_project_quote(request)

        self.assertEqual(quote["amount"], quote["recommendedMinimum"])
        self.assertLess(quote["amount"], 275_000)
        prompt = _build_prompt(request)
        self.assertNotIn("275000", prompt)
        self.assertNotIn("budgetMin", prompt)

    def test_quote_rejects_unpriceable_scope(self):
        with self.assertRaisesRegex(ValueError, "scopeDetails"):
            estimate_project_quote(
                ProjectQuoteRequest(
                    project={"budgetMin": 500, "budgetMax": 300_000},
                    brief={
                        "mainGoal": "I want to make a mobile website",
                        "targetUsers": ["customers"],
                        "coreFeatures": ["website"],
                        "platforms": ["website"],
                        "solutionType": "responsive website",
                    },
                )
            )


if __name__ == "__main__":
    unittest.main()
