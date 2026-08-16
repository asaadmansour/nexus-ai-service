"""Evaluate an immutable implementation snapshot after secret-free verification."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from app.agents.submission_evaluation import evaluate_submission

RESULT_MARKER = "NEXUS_EVALUATION_RESULT:"
AUDIT_MARKER = "NEXUS_EVALUATION_AUDIT:"
MAX_INSPECTED_FILES = int(os.getenv("IMPLEMENTATION_MAX_INSPECTED_FILES", "100"))
MAX_SOURCE_CHARS = int(os.getenv("IMPLEMENTATION_MAX_SOURCE_CHARS", "350000"))
MAX_FILE_CHARS = int(os.getenv("IMPLEMENTATION_MAX_FILE_CHARS", "30000"))
MAX_VERIFICATION_OUTPUT_CHARS = int(
    os.getenv("IMPLEMENTATION_MAX_VERIFICATION_OUTPUT_CHARS", "8000")
)
MAX_MANIFEST_SAMPLE = 500


def main() -> int:
    try:
        request = _request()
        github = _read_json(Path("/workspace/snapshot-evidence/github.json"))
        verification = _read_json(
            Path("/workspace/verification-evidence/verification.json")
        )
        inspection = _inspection(Path("/workspace/source"), github, verification)
        submission = request.setdefault("submission", {})
        submission["inspection"] = inspection
        submission["commitSha"] = github.get("commitSha")

        result = evaluate_submission(request)
        serialized = json.dumps(result, sort_keys=True, separators=(",", ":"))
        audit = {
            "schemaVersion": 1,
            "commitSha": github.get("commitSha"),
            "baseCommitSha": github.get("baseCommitSha"),
            "snapshotVerified": github.get("snapshotVerified") is True,
            "sourceManifestHash": github.get("sourceManifestHash"),
            "archive": github.get("archive"),
            "pullRequest": github.get("pullRequest"),
            "githubChecks": github.get("githubChecks"),
            "verification": verification,
            "inspectionCoverage": inspection.get("coverage"),
            "evaluationInputHash": _hash_json(request),
            "verdictSha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "summaryMarkdown": _summary(result, github, verification, request),
        }
        output = Path("/workspace/output")
        output.mkdir(parents=True, exist_ok=True)
        (output / "verdict.json").write_text(serialized, encoding="utf-8")
        (output / "summary.md").write_text(audit["summaryMarkdown"], encoding="utf-8")
        print(RESULT_MARKER + base64.b64encode(serialized.encode()).decode())
        encoded_audit = json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()
        print(AUDIT_MARKER + base64.b64encode(encoded_audit).decode())
        return 0
    except Exception as exc:
        print(f"Implementation evaluation failed: {exc}", file=sys.stderr)
        return 1


def _request() -> dict[str, Any]:
    encoded = os.getenv("EVALUATION_REQUEST_B64", "")
    if not encoded:
        raise RuntimeError("EVALUATION_REQUEST_B64 is required")
    value = json.loads(base64.b64decode(encoded, validate=True))
    if not isinstance(value, dict):
        raise RuntimeError("Evaluation request must be an object")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid evidence file: {path.name}")
    return value


def _inspection(source: Path, github: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
    manifest = github.get("sourceManifest") or []
    manifest_paths = [item.get("path") for item in manifest if isinstance(item, dict) and isinstance(item.get("path"), str)]
    changed_items = [
        item for item in github.get("changedFiles") or [] if isinstance(item, dict)
    ]
    changed = [
        item.get("path")
        for item in changed_items
        if isinstance(item.get("path"), str)
    ]
    preferred = changed + [
        path
        for path in manifest_paths
        if Path(path).name.lower() in {"readme.md", "package.json", "pyproject.toml", "requirements.txt", "dockerfile"}
    ] + manifest_paths
    selected: list[str] = []
    for relative in preferred:
        if relative not in selected and len(selected) < MAX_INSPECTED_FILES:
            selected.append(relative)

    excerpts: list[dict[str, Any]] = []
    consumed = 0
    for relative in selected:
        path = source / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 2_000_000:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        remaining = MAX_SOURCE_CHARS - consumed
        if remaining <= 0:
            break
        text = raw[: min(MAX_FILE_CHARS, remaining)]
        numbered = "\n".join(f"{index + 1}: {line}" for index, line in enumerate(text.splitlines()))
        consumed += len(numbered)
        excerpts.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "content": numbered,
                "truncated": len(raw) > len(text),
            }
        )
    changed_set = set(changed)
    inspected_set = {item["path"] for item in excerpts}
    removed_with_patch = {
        item.get("path")
        for item in changed_items
        if item.get("status") == "removed"
        and isinstance(item.get("path"), str)
        and isinstance(item.get("patch"), str)
    }
    covered_changed = inspected_set | removed_with_patch
    changed_coverage = (
        1.0
        if not changed_set
        else len(changed_set & covered_changed) / len(changed_set)
    )
    complete = (
        github.get("snapshotVerified") is True
        and verification.get("complete") is True
        and changed_coverage == 1.0
        and not github.get("diffTruncated")
        and not github.get("changedFilesTruncated")
        and not (github.get("githubChecks") or {}).get("truncated")
    )
    manifest_sample = [
        {"path": item.get("path"), "sizeBytes": item.get("sizeBytes")}
        for item in manifest[:MAX_MANIFEST_SAMPLE]
        if isinstance(item, dict)
    ]
    return {
        "schemaVersion": 1,
        "sourceInspected": True,
        "snapshotVerified": github.get("snapshotVerified") is True,
        "verificationComplete": verification.get("complete") is True,
        "complete": complete,
        "commitSha": github.get("commitSha"),
        "baseCommitSha": github.get("baseCommitSha"),
        "pullRequest": github.get("pullRequest"),
        "changedFiles": github.get("changedFiles"),
        "diff": github.get("diff"),
        "diffTruncated": github.get("diffTruncated"),
        "changedFilesTruncated": github.get("changedFilesTruncated"),
        "githubChecks": github.get("githubChecks"),
        "verification": _verification_for_model(verification),
        "sourceManifestSample": manifest_sample,
        "sourceManifestTruncated": len(manifest) > len(manifest_sample),
        "sourceExcerpts": excerpts,
        "coverage": {
            "manifestFiles": len(manifest_paths),
            "changedFiles": len(changed_set),
            "inspectedFiles": len(excerpts),
            "changedFileCoverage": changed_coverage,
            "sourceChars": consumed,
        },
    }


def _verification_for_model(verification: dict[str, Any]) -> dict[str, Any]:
    """Keep full evidence in the audit bundle while bounding the model context."""
    results: list[dict[str, Any]] = []
    for item in verification.get("results") or []:
        if not isinstance(item, dict):
            continue
        bounded = dict(item)
        output = str(bounded.get("output") or "")
        if len(output) > MAX_VERIFICATION_OUTPUT_CHARS:
            bounded["output"] = output[-MAX_VERIFICATION_OUTPUT_CHARS:]
            bounded["outputTruncatedForEvaluation"] = True
        results.append(bounded)
    return {**verification, "results": results}


def _summary(
    result: dict[str, Any],
    github: dict[str, Any],
    verification: dict[str, Any],
    request: dict[str, Any],
) -> str:
    unmet = [item.get("criterion") for item in result.get("rubric") or [] if isinstance(item, dict) and not item.get("met")]
    history = [
        item
        for item in request.get("evaluationHistory") or []
        if isinstance(item, dict)
    ]
    prior_lines = [
        "- "
        + f"{item.get('commitSha') or 'no commit'}: "
        + f"{item.get('recommendation') or 'no recommendation'}; "
        + f"unmet: {', '.join(str(value) for value in item.get('unmetCriteria') or []) or 'none'}"
        for item in history
    ]
    return "\n".join(
        [
            "# Implementation evaluation",
            "",
            f"- Commit: {github.get('commitSha', 'unknown')}",
            f"- Passed: {bool(result.get('passed'))}",
            f"- Score: {result.get('score', 'unknown')}",
            f"- Human review required: {bool(result.get('requiresHumanReview'))}",
            f"- Verification failures: {verification.get('commandsFailed', 'unknown')}",
            "",
            "## Unmet criteria",
            *([f"- {item}" for item in unmet] or ["- None"]),
            "",
            "## Revision notes",
            str(result.get("revisionNotes") or "None"),
            "",
            "## Prior verdicts",
            *(prior_lines or ["- None"]),
            "",
        ]
    )


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
