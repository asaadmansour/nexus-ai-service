"""Run one planning evaluation inside an ephemeral sandbox container."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

from app.agents.planning_submission_evaluation import evaluate_submission

RESULT_MARKER = "NEXUS_EVALUATION_RESULT:"
AUDIT_MARKER = "NEXUS_EVALUATION_AUDIT:"


def main() -> int:
    encoded = os.getenv("EVALUATION_REQUEST_B64", "")
    if not encoded:
        print("EVALUATION_REQUEST_B64 is required", file=sys.stderr)
        return 2
    try:
        request = json.loads(base64.b64decode(encoded, validate=True))
        output_dir = Path("/workspace/output")
        if output_dir.exists():
            (output_dir / "summary.md").write_text(
                _prior_summary(request.get("previousVerdict")), encoding="utf-8"
            )
        result = evaluate_submission(request)
        serialized = json.dumps(result, sort_keys=True, separators=(",", ":"))
        summary = _summary(request.get("previousVerdict"), result)
        if output_dir.exists():
            (output_dir / "verdict.json").write_text(serialized, encoding="utf-8")
            (output_dir / "summary.md").write_text(
                summary, encoding="utf-8"
            )
        print(RESULT_MARKER + base64.b64encode(serialized.encode()).decode())
        audit = json.dumps(
            {
                "schemaVersion": 1,
                "summaryMarkdown": summary,
                "verdictSha256": hashlib.sha256(serialized.encode()).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        print(AUDIT_MARKER + base64.b64encode(audit.encode()).decode())
        return 0
    except Exception as exc:
        print(f"Planning evaluation failed: {exc}", file=sys.stderr)
        return 1


def _prior_summary(previous: object) -> str:
    if not isinstance(previous, dict) or not previous:
        return "# Planning evaluation history\n\nNo prior verdict.\n"
    return "\n".join(
        [
            "# Planning evaluation history",
            "",
            "## Prior verdict",
            f"- Recommendation: {previous.get('recommendation', 'unknown')}",
            f"- Score: {previous.get('score', 'unknown')}",
            f"- Input hash: {previous.get('evaluationInputHash', 'unknown')}",
            "",
            str(previous.get("summary") or ""),
            "",
            "### Prior open issues",
            *_issue_lines(previous.get("openIssues")),
            "",
        ]
    )


def _summary(previous: object, result: dict) -> str:
    lines = [
        _prior_summary(previous).rstrip(),
        "",
        "## Current verdict",
        f"- Recommendation: {result.get('recommendation', 'unknown')}",
        f"- Score: {result.get('score', 'unknown')}",
        f"- Input hash: {result.get('evaluationInputHash', 'unknown')}",
        f"- Reused deterministically: {bool(result.get('reused'))}",
        "",
        str(result.get("summary") or ""),
        "",
        "### Current open issues",
        *_issue_lines(result.get("openIssues")),
        "",
        "### Resolved prior issues",
        *_value_lines(result.get("resolvedIssues")),
        "",
        "### Regressions",
        *_value_lines(result.get("regressions")),
    ]
    return "\n".join(lines) + "\n"


def _issue_lines(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["- None"]
    return [
        f"- [{issue.get('criterionKey', 'unknown')}] {issue.get('message', '')}"
        for issue in value
        if isinstance(issue, dict)
    ] or ["- None"]


def _value_lines(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["- None"]
    return [f"- {item}" for item in value]


if __name__ == "__main__":
    raise SystemExit(main())
