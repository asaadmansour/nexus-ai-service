import os
import unittest
from unittest.mock import patch

from main import (
    health,
    is_ai_provider_configured,
    is_figma_provider_configured,
    is_figma_smoke_configured,
)


class AiServiceHealthTests(unittest.TestCase):
    def test_provider_is_not_ready_without_a_real_key(self):
        for value in (None, "", "change-me"):
            env = {} if value is None else {"GEMINI_API_KEY": value}
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                self.assertFalse(is_ai_provider_configured())

    def test_provider_is_ready_with_a_non_placeholder_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            self.assertTrue(is_ai_provider_configured())

    def test_optional_figma_does_not_make_gemini_health_unavailable(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            result = health()
            self.assertIsInstance(result, dict)
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["aiProviderConfigured"])
            self.assertFalse(result["checks"]["figma"])

    def test_figma_provider_requires_a_non_placeholder_token(self):
        for value, expected in ((None, False), ("change-me", False), ("token", True)):
            env = {} if value is None else {"FIGMA_ACCESS_TOKEN": value}
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                self.assertEqual(is_figma_provider_configured(), expected)

    def test_figma_smoke_requires_a_figma_https_url(self):
        for value, expected in (
            ("", False),
            ("https://example.com/design/file", False),
            ("https://www.figma.com/design/file-key/name", True),
        ):
            with self.subTest(value=value), patch.dict(
                os.environ, {"FIGMA_SMOKE_FILE_URL": value}, clear=True
            ):
                self.assertEqual(is_figma_smoke_configured(), expected)


if __name__ == "__main__":
    unittest.main()
