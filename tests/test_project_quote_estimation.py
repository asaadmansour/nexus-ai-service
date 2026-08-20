import unittest
from unittest.mock import patch

from app.agents.project_quote_estimation import ProjectQuoteRequest, _fallback_quote


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


if __name__ == "__main__":
    unittest.main()
