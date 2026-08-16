"""Acquire an immutable GitHub source snapshot before untrusted verification.

This runner is intended for a Kubernetes init container. It receives the
GitHub token, downloads one exact commit, writes only source/evidence to shared
volumes, and exits before any repository code is executed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MAX_ARCHIVE_BYTES = int(os.getenv("IMPLEMENTATION_MAX_ARCHIVE_BYTES", str(250 * 1024 * 1024)))
MAX_EXTRACTED_BYTES = int(
    os.getenv("IMPLEMENTATION_MAX_EXTRACTED_BYTES", str(500 * 1024 * 1024))
)
MAX_ARCHIVE_MEMBERS = 50_000
MAX_API_BYTES = 8 * 1024 * 1024
MAX_DIFF_BYTES = 750 * 1024
MAX_PATCH_CHARS = 12_000


class SnapshotError(RuntimeError):
    pass


def _allowed_request_host(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    configured = urllib.parse.urlparse(os.getenv("GITHUB_API_URL", ""))
    allowed_hosts = {
        "api.github.com",
        "github.com",
        "codeload.github.com",
        configured.hostname,
    }
    if parsed.scheme != "https" or not parsed.hostname or parsed.hostname not in allowed_hosts:
        raise SnapshotError("Refusing an unexpected GitHub API host")
    return parsed.hostname


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        _allowed_request_host(req.full_url)
        _allowed_request_host(newurl)
        return redirected


_OPENER = urllib.request.build_opener(_SafeRedirectHandler())


def main() -> int:
    try:
        owner = _required("GITHUB_REPOSITORY_OWNER")
        repository = _required("GITHUB_REPOSITORY_NAME")
        expected_sha = _required("GITHUB_COMMIT_SHA").lower()
        token = _required("GITHUB_TOKEN")
        pr_number = os.getenv("GITHUB_PULL_REQUEST_NUMBER", "").strip()
        api_url = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")

        _validate_slug(owner, "owner")
        _validate_slug(repository, "repository")
        if not _is_sha(expected_sha):
            raise SnapshotError("GITHUB_COMMIT_SHA must be a full hexadecimal commit SHA")

        pull: dict[str, Any] | None = None
        if pr_number:
            if not pr_number.isdigit() or int(pr_number) <= 0:
                raise SnapshotError("Invalid pull-request number")
            pull = _json_request(
                f"{api_url}/repos/{_quote(owner)}/{_quote(repository)}/pulls/{pr_number}",
                token,
            )
            head_sha = str(((pull.get("head") or {}).get("sha") or "")).lower()
            if head_sha != expected_sha:
                raise SnapshotError(
                    "The pull-request head changed after the evaluation target was resolved"
                )

        commit = _json_request(
            f"{api_url}/repos/{_quote(owner)}/{_quote(repository)}/commits/{expected_sha}",
            token,
        )
        resolved_sha = str(commit.get("sha") or "").lower()
        if resolved_sha != expected_sha:
            raise SnapshotError("GitHub resolved a different commit than requested")

        checks = _json_request(
            f"{api_url}/repos/{_quote(owner)}/{_quote(repository)}/commits/{expected_sha}/check-runs?per_page=100",
            token,
        )
        status = _json_request(
            f"{api_url}/repos/{_quote(owner)}/{_quote(repository)}/commits/{expected_sha}/status",
            token,
        )
        diff = ""
        diff_truncated = False
        pull_files: list[dict[str, Any]] = []
        pull_files_truncated = False
        if pull:
            diff, diff_truncated = _text_request(
                f"{api_url}/repos/{_quote(owner)}/{_quote(repository)}/pulls/{pr_number}",
                token,
                "application/vnd.github.diff",
                MAX_DIFF_BYTES,
            )
            pull_files, pull_files_truncated = _paginated_list_request(
                f"{api_url}/repos/{_quote(owner)}/{_quote(repository)}/pulls/{pr_number}/files",
                token,
                max_pages=10,
            )

        source_dir = Path("/workspace/source")
        evidence_dir = Path("/workspace/evidence")
        snapshot_work_dir = Path("/workspace/snapshot")
        source_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        snapshot_work_dir.mkdir(parents=True, exist_ok=True)
        _clear_directory(source_dir)
        with tempfile.TemporaryDirectory(
            prefix="nexus-github-snapshot-", dir=snapshot_work_dir
        ) as directory:
            archive_path = Path(directory) / "source.tar.gz"
            archive_sha, archive_size = _download(
                f"{api_url}/repos/{_quote(owner)}/{_quote(repository)}/tarball/{expected_sha}",
                token,
                archive_path,
                MAX_ARCHIVE_BYTES,
            )
            _extract_archive(archive_path, source_dir, Path(directory))

        files = _source_manifest(source_dir)
        metadata = {
            "schemaVersion": 1,
            "repository": {"owner": owner, "name": repository},
            "commitSha": expected_sha,
            "baseCommitSha": str(((pull or {}).get("base") or {}).get("sha") or "") or None,
            "pullRequest": _pull_summary(pull, pr_number),
            "commit": _commit_summary(commit),
            "changedFiles": _changed_files(pull_files if pull else commit.get("files") or []),
            "changedFilesTruncated": pull_files_truncated
            if pull
            else len(commit.get("files") or []) >= 300,
            "diff": diff,
            "diffTruncated": diff_truncated,
            "githubChecks": _checks_summary(checks, status),
            "archive": {"sha256": archive_sha, "sizeBytes": archive_size},
            "sourceManifest": files,
            "sourceManifestHash": _hash_json(files),
            "snapshotVerified": True,
        }
        (evidence_dir / "github.json").write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"Snapshot acquired at {expected_sha} ({len(files)} files)")
        return 0
    except Exception as exc:
        print(f"GitHub snapshot failed: {exc}", file=sys.stderr)
        return 1


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SnapshotError(f"{name} is required")
    return value


def _validate_slug(value: str, label: str) -> None:
    if not value or len(value) > 160 or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        for character in value
    ):
        raise SnapshotError(f"Invalid GitHub {label}")


def _is_sha(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _request(url: str, token: str, accept: str):
    _allowed_request_host(url)
    return _OPENER.open(
        urllib.request.Request(
            url,
            headers={
                "Accept": accept,
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "nexus-ai-implementation-evaluator",
            },
        ),
        timeout=60,
    )


def _read_limited(response, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = response.read(min(1024 * 1024, limit + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise SnapshotError("GitHub response exceeded the configured size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _json_request(url: str, token: str) -> dict[str, Any]:
    try:
        with _request(url, token, "application/vnd.github+json") as response:
            value = json.loads(_read_limited(response, MAX_API_BYTES))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"GitHub API request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotError("GitHub API returned an unexpected payload")
    return value


def _paginated_list_request(
    url: str, token: str, max_pages: int
) -> tuple[list[dict[str, Any]], bool]:
    collected: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        separator = "&" if "?" in url else "?"
        try:
            with _request(
                f"{url}{separator}per_page=100&page={page}",
                token,
                "application/vnd.github+json",
            ) as response:
                value = json.loads(_read_limited(response, MAX_API_BYTES))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"GitHub paginated request failed: {exc}") from exc
        if not isinstance(value, list):
            raise SnapshotError("GitHub API returned an unexpected list payload")
        page_items = [item for item in value if isinstance(item, dict)]
        collected.extend(page_items)
        if len(value) < 100:
            return collected, False
    return collected, True


def _text_request(
    url: str, token: str, accept: str, limit: int
) -> tuple[str, bool]:
    try:
        with _request(url, token, accept) as response:
            value = response.read(limit + 1)
            return (
                value[:limit].decode("utf-8", errors="replace"),
                len(value) > limit,
            )
    except urllib.error.URLError as exc:
        raise SnapshotError(f"GitHub diff request failed: {exc}") from exc


def _download(url: str, token: str, target: Path, limit: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with _request(url, token, "application/vnd.github+json") as response, target.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise SnapshotError("Repository archive exceeded the configured size limit")
                digest.update(chunk)
                output.write(chunk)
    except urllib.error.URLError as exc:
        raise SnapshotError(f"Repository archive download failed: {exc}") from exc
    return digest.hexdigest(), size


def _clear_directory(directory: Path) -> None:
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _extract_archive(
    archive_path: Path, destination: Path, temporary_root: Path
) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise SnapshotError("Repository archive contains too many entries")
        extracted_bytes = sum(member.size for member in members if member.isfile())
        if extracted_bytes > MAX_EXTRACTED_BYTES:
            raise SnapshotError(
                "Repository archive exceeded the extracted-size limit"
            )
        roots = {member.name.split("/", 1)[0] for member in members if member.name}
        if len(roots) != 1:
            raise SnapshotError("GitHub archive has an unexpected layout")
        root = next(iter(roots))
        with tempfile.TemporaryDirectory(
            prefix="nexus-extract-", dir=temporary_root
        ) as directory:
            temporary = Path(directory)
            archive.extractall(temporary, filter="data")
            extracted_root = temporary / root
            if not extracted_root.is_dir():
                raise SnapshotError("GitHub archive did not contain a repository directory")
            for child in extracted_root.iterdir():
                shutil.move(str(child), destination / child.name)


def _source_manifest(source_dir: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source_dir).as_posix()
        size = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append({"path": relative, "sizeBytes": size, "sha256": digest.hexdigest()})
        if len(files) > 50_000:
            raise SnapshotError("Repository contains too many files")
    return files


def _pull_summary(pull: dict[str, Any] | None, number: str) -> dict[str, Any] | None:
    if not pull:
        return None
    return {
        "number": int(number),
        "url": pull.get("html_url"),
        "state": pull.get("state"),
        "draft": pull.get("draft"),
        "mergeable": pull.get("mergeable"),
        "mergeableState": pull.get("mergeable_state"),
        "headRef": (pull.get("head") or {}).get("ref"),
        "headSha": (pull.get("head") or {}).get("sha"),
        "baseRef": (pull.get("base") or {}).get("ref"),
        "baseSha": (pull.get("base") or {}).get("sha"),
        "changedFiles": pull.get("changed_files"),
        "additions": pull.get("additions"),
        "deletions": pull.get("deletions"),
    }


def _commit_summary(commit: dict[str, Any]) -> dict[str, Any]:
    details = commit.get("commit") or {}
    return {
        "sha": commit.get("sha"),
        "url": commit.get("html_url"),
        "message": details.get("message"),
        "author": (details.get("author") or {}).get("name"),
        "authoredAt": (details.get("author") or {}).get("date"),
        "parents": [item.get("sha") for item in commit.get("parents") or [] if isinstance(item, dict)],
    }


def _changed_files(files: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": item.get("filename"),
            "previousPath": item.get("previous_filename"),
            "status": item.get("status"),
            "additions": item.get("additions"),
            "deletions": item.get("deletions"),
            "changes": item.get("changes"),
            "patch": item.get("patch")[:MAX_PATCH_CHARS]
            if isinstance(item.get("patch"), str)
            else None,
            "patchTruncated": isinstance(item.get("patch"), str)
            and len(item.get("patch")) > MAX_PATCH_CHARS,
        }
        for item in files
        if isinstance(item, dict)
    ]


def _checks_summary(checks: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    check_runs = [
        item for item in checks.get("check_runs") or [] if isinstance(item, dict)
    ]
    return {
        "checkRuns": [
            {
                "name": item.get("name"),
                "status": item.get("status"),
                "conclusion": item.get("conclusion"),
                "url": item.get("html_url"),
            }
            for item in check_runs
        ],
        "truncated": int(checks.get("total_count") or 0) > len(check_runs),
        "combinedStatus": status.get("state"),
        "statuses": [
            {
                "context": item.get("context"),
                "state": item.get("state"),
                "description": item.get("description"),
                "url": item.get("target_url"),
            }
            for item in status.get("statuses") or []
            if isinstance(item, dict)
        ],
    }


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
