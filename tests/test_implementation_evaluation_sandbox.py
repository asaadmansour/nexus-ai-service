import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.runners.implementation_evaluation import (
    MAX_VERIFICATION_OUTPUT_CHARS,
    _inspection,
    _summary,
    _verification_for_model,
)
from app.runners.implementation_verification import (
    MAX_OUTPUT_CHARS,
    _run,
    _safe_env,
    _verify_static_web,
)
from app.runners.github_snapshot import (
    SnapshotError,
    _allowed_request_host,
    _extract_archive,
)


class ImplementationEvaluationSandboxTests(unittest.TestCase):
    def test_verifier_environment_drops_platform_secrets(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "github-secret",
                "GEMINI_API_KEY": "gemini-secret",
                "DATABASE_URL": "database-secret",
                "PATH": os.environ.get("PATH", ""),
            },
        ):
            env = _safe_env(Path(directory))

        self.assertNotIn("GITHUB_TOKEN", env)
        self.assertNotIn("GEMINI_API_KEY", env)
        self.assertNotIn("DATABASE_URL", env)
        self.assertIn("PATH", env)

    def test_inspection_requires_all_changed_files_and_complete_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "app.py").write_text("print('ok')\n", encoding="utf-8")
            github = {
                "commitSha": "a" * 40,
                "snapshotVerified": True,
                "diffTruncated": False,
                "changedFiles": [{"path": "app.py"}],
                "sourceManifest": [{"path": "app.py"}],
            }
            verification = {"complete": True, "results": []}

            result = _inspection(source, github, verification)

        self.assertTrue(result["complete"])
        self.assertEqual(result["coverage"]["changedFileCoverage"], 1.0)
        self.assertEqual(result["sourceExcerpts"][0]["path"], "app.py")

    def test_snapshot_rejects_redirects_outside_the_github_allowlist(self):
        with self.assertRaises(SnapshotError):
            _allowed_request_host("https://attacker.example/archive.tar.gz")

    def test_snapshot_rejects_archive_expansion_over_the_disk_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "source.tar.gz"
            destination = root / "source"
            destination.mkdir()
            with tarfile.open(archive_path, "w:gz") as archive:
                member = tarfile.TarInfo("repository/large.bin")
                member.size = 10
                archive.addfile(member, io.BytesIO(b"x" * member.size))

            with patch(
                "app.runners.github_snapshot.MAX_EXTRACTED_BYTES", 5
            ), self.assertRaises(SnapshotError):
                _extract_archive(archive_path, destination, root)

    def test_model_context_bounds_full_verification_logs(self):
        report = {
            "complete": True,
            "results": [{"name": "tests", "output": "x" * 20_000}],
        }

        bounded = _verification_for_model(report)

        self.assertEqual(
            len(bounded["results"][0]["output"]),
            MAX_VERIFICATION_OUTPUT_CHARS,
        )
        self.assertTrue(
            bounded["results"][0]["outputTruncatedForEvaluation"]
        )

    def test_verifier_bounds_process_output_while_the_command_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = _run(
                ".",
                Path(directory),
                "test",
                "large-output",
                [sys.executable, "-c", "print('x' * 200000)"],
                10,
            )

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["outputTruncated"])
        self.assertLessEqual(len(result["output"]), MAX_OUTPUT_CHARS)

    def test_static_site_gets_proportionate_verification_without_a_test_suite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "styles.css").write_text("body { color: black; }", encoding="utf-8")
            (root / "index.html").write_text(
                '<!doctype html><link href="styles.css" rel="stylesheet"><h1>Hello World</h1>',
                encoding="utf-8",
            )

            result = _verify_static_web(root)

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["category"], "build")
        self.assertIn("Validated 1 HTML", result["output"])

    def test_static_site_verification_reports_broken_local_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text(
                '<!doctype html><script src="missing.js"></script>',
                encoding="utf-8",
            )

            result = _verify_static_web(root)

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "failed")
        self.assertIn("missing local asset", result["output"])

    def test_persisted_summary_carries_prior_verdicts(self):
        summary = _summary(
            {"passed": True, "score": 90, "rubric": [], "revisionNotes": ""},
            {"commitSha": "b" * 40},
            {"commandsFailed": 0},
            {
                "evaluationHistory": [
                    {
                        "commitSha": "a" * 40,
                        "recommendation": "changes_requested",
                        "unmetCriteria": ["Contract tests pass"],
                    }
                ]
            },
        )

        self.assertIn("## Prior verdicts", summary)
        self.assertIn("changes_requested", summary)
        self.assertIn("Contract tests pass", summary)


if __name__ == "__main__":
    unittest.main()
