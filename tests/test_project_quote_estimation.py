import unittest

from app.agents.project_quote_estimation import (
    ProjectQuoteEstimationError,
    ProjectQuoteRequest,
    ProjectQuoteResponse,
    RoleEstimate,
    _build_prompt,
    _fallback_quote,
    _is_minimal_website_scope,
    _scope_tier,
    _normalize_quote_response,
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

    def test_small_marketing_site_does_not_jump_to_standard_app_hours(self):
        brief = {
            "mainGoal": "Present services and collect enquiries",
            "targetUsers": ["potential customers"],
            "coreFeatures": ["service pages", "contact form", "testimonials"],
            "platforms": ["website"],
            "solutionType": "multi-page marketing website",
            "scopeDetails": "five pages: home, services, about, testimonials, contact",
            "integrations": "none",
            "adminNeeds": "no admin dashboard",
            "deliverables": ["working website", "source code", "live link"],
        }
        self.assertEqual(_scope_tier(brief), "small")
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
        self.assertEqual(implementation["hoursEach"], 32)
        self.assertLess(quote["amount"], 50_000)

    def test_fixed_package_changes_with_confirmed_scope(self):
        landing_request = self.request(500_000)
        landing_request.brief.update(
            {
                "solutionType": "single landing page",
                "scopeDetails": "one page with five static sections",
                "integrations": "none",
                "adminNeeds": "no admin dashboard",
            }
        )
        landing = _fallback_quote(landing_request, "fallback")
        application_request = ProjectQuoteRequest(
            project={"budgetMin": 500, "budgetMax": 500_000, "currency": "EGP"},
            brief={
                "mainGoal": "Run customer orders",
                "targetUsers": ["customers", "staff"],
                "coreFeatures": [
                    "accounts",
                    "catalog",
                    "checkout",
                    "orders",
                    "reporting",
                    "notifications",
                ],
                "platforms": ["web app"],
                "solutionType": "custom web app",
                "scopeDetails": "twelve screens for customers and staff",
                "integrations": ["Stripe", "email"],
                "adminNeeds": "admin dashboard for orders and users",
                "deliverables": ["working application", "source code"],
            },
        )
        application = _fallback_quote(application_request, "fallback")

        self.assertGreater(application["amount"], landing["amount"] * 3)

    def test_detailed_planning_does_not_inflate_a_focused_web_app(self):
        request = ProjectQuoteRequest(
            project={
                "budgetMin": 500,
                "budgetMax": 100_000,
                "currency": "EGP",
                "deadline": "2027-10-10T00:00:00.000Z",
            },
            brief={
                "mainGoal": "Manage appointment booking and a daily clinic queue",
                "targetUsers": ["patients", "reception staff", "clinic managers"],
                "coreFeatures": [
                    "patients book appointments",
                    "patients reschedule appointments",
                    "reception staff manage daily queue",
                    "clinic managers review appointment activity",
                ],
                "platforms": ["website"],
                "solutionType": "web app",
                "scopeDetails": (
                    "One-location booking, reception queue, and manager reporting workflows"
                ),
                "integrations": ["none"],
                "adminNeeds": "no admin dashboard",
                "deliverables": [
                    "working website",
                    "source code",
                    "deployment help",
                ],
                "suggestedTeamSize": 2,
                "requirementProfile": {"complexity": "complex"},
            },
        )

        self.assertEqual(_scope_tier(request.brief), "standard")
        quote = _fallback_quote(request, "fallback")
        self.assertGreaterEqual(quote["amount"], 35_000)
        self.assertLessEqual(quote["amount"], 40_000)

    def test_fallback_rates_are_expressed_in_the_project_currency(self):
        request = self.request(10_000)
        request.project["currency"] = "USD"
        request.brief.update(
            {
                "solutionType": "single landing page",
                "scopeDetails": "one page with one heading",
                "integrations": "none",
                "adminNeeds": "no admin dashboard",
            }
        )
        quote = _fallback_quote(request, "fallback")

        self.assertEqual(quote["currency"], "USD")
        self.assertLess(quote["amount"], 1_000)
        self.assertTrue(
            all(row["hourlyRate"] < 100 for row in quote["roleEstimates"])
        )

    def test_ai_quote_with_the_wrong_currency_is_rejected(self):
        role_estimates = [
            RoleEstimate(
                roleKey=role,
                people=1,
                hoursEach=hours,
                hourlyRate=rate,
                subtotal=hours * rate,
            )
            for role, hours, rate in (
                ("principal_reviewer", 2, 15),
                ("architect", 2, 14),
                ("ui_ux", 2, 10),
                ("implementation", 8, 8),
            )
        ]
        quote = ProjectQuoteResponse(
            amount=1,
            recommendedMinimum=1,
            roleEstimates=role_estimates,
            currency="EGP",
            complexity="low",
            rationale="test",
        )
        request = self.request(10_000)
        request.project["currency"] = "USD"
        request.brief.update(
            {
                "solutionType": "single landing page",
                "scopeDetails": "one page with one heading",
                "integrations": "none",
                "adminNeeds": "no admin dashboard",
            }
        )

        with self.assertRaises(ProjectQuoteEstimationError):
            _normalize_quote_response(quote, request, 10_000)

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
