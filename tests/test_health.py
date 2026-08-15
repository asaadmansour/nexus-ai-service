import os
import unittest
from unittest.mock import patch

from main import is_ai_provider_configured


class AiServiceHealthTests(unittest.TestCase):
    def test_provider_is_not_ready_without_a_real_key(self):
        for value in (None, "", "change-me"):
            env = {} if value is None else {"GEMINI_API_KEY": value}
            with self.subTest(value=value), patch.dict(os.environ, env, clear=True):
                self.assertFalse(is_ai_provider_configured())

    def test_provider_is_ready_with_a_non_placeholder_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True):
            self.assertTrue(is_ai_provider_configured())


if __name__ == "__main__":
    unittest.main()
