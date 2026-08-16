"""Secure artifact acquisition for isolated planning evaluations.

Submitted URLs are untrusted.  This module snapshots their bytes, hashes the
snapshot, blocks private-network access, and prepares supported documents and
images for Gemini.  Figma files are inspected through the read-only REST API
and their top-level frames are rendered for multimodal review.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import mimetypes
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

MAX_URLS = int(os.getenv("EVALUATION_MAX_ARTIFACTS", "30"))
MAX_FILE_BYTES = int(os.getenv("EVALUATION_MAX_FILE_BYTES", str(50 * 1024 * 1024)))
MAX_TOTAL_BYTES = int(os.getenv("EVALUATION_MAX_TOTAL_BYTES", str(100 * 1024 * 1024)))
MAX_TEXT_BYTES = int(os.getenv("EVALUATION_MAX_TEXT_BYTES", str(250 * 1024)))
MAX_EXTRACTED_TEXT_CHARS = int(
    os.getenv("EVALUATION_MAX_EXTRACTED_TEXT_CHARS", "1000000")
)
MAX_FIGMA_FRAMES = int(os.getenv("EVALUATION_MAX_FIGMA_FRAMES", "20"))
MAX_FIGMA_STRUCTURE_FRAMES = int(
    os.getenv("EVALUATION_MAX_FIGMA_STRUCTURE_FRAMES", "500")
)
MAX_FIGMA_PAGES = int(os.getenv("EVALUATION_MAX_FIGMA_PAGES", "100"))
ALLOWED_MEDIA_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
    "image/webp",
}
ALLOWED_TEXT_TYPES = {
    "application/json",
    "application/yaml",
    "application/x-yaml",
    "application/xml",
    "text/html",
    "text/markdown",
    "text/plain",
    "text/xml",
    "text/yaml",
}
FIGMA_URL_RE = re.compile(
    r"^/(?:design|file|proto|board|slides)/([^/?#]+)", re.IGNORECASE
)


class ArtifactInspectionError(RuntimeError):
    pass


def inspect_artifacts(
    request: Dict[str, Any], workspace: Path, reuse_previous_figma: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Return an immutable manifest, Gemini media parts, and extracted text."""

    workspace.mkdir(parents=True, exist_ok=True)
    url_requirements = _collect_urls(request)
    artifacts: List[Dict[str, Any]] = []
    media: List[Dict[str, Any]] = []
    extracted_text: List[str] = []
    consumed = 0
    previous_artifacts = (
        ((request.get("previousVerdict") or {}).get("artifactManifest") or {}).get(
            "artifacts"
        )
        or []
    )

    for index, (url, requirement_keys) in enumerate(url_requirements[:MAX_URLS]):
        artifact_id = f"artifact-{index + 1}-{hashlib.sha256(url.encode()).hexdigest()[:10]}"
        try:
            if _figma_key(url):
                figma_artifacts, figma_media, figma_text = _inspect_figma(
                    url,
                    artifact_id,
                    requirement_keys,
                    workspace,
                    previous_artifacts if reuse_previous_figma else [],
                )
                new_bytes = sum(int(item.get("sizeBytes") or 0) for item in figma_artifacts)
                if consumed + new_bytes > MAX_TOTAL_BYTES:
                    raise ArtifactInspectionError("Total artifact size limit exceeded")
                consumed += new_bytes
                artifacts.extend(figma_artifacts)
                media.extend(figma_media)
                extracted_text.append(figma_text)
                continue

            data, content_type, final_url = _download(url, MAX_FILE_BYTES)
            consumed += len(data)
            if consumed > MAX_TOTAL_BYTES:
                raise ArtifactInspectionError("Total artifact size limit exceeded")
            content_type = _normalize_content_type(content_type, final_url)
            path = workspace / f"{artifact_id}{_extension(content_type, final_url)}"
            path.write_bytes(data)
            artifact = _artifact_record(
                artifact_id,
                url,
                final_url,
                requirement_keys,
                content_type,
                data,
            )
            if content_type in ALLOWED_MEDIA_TYPES:
                media.append(
                    {
                        "artifactId": artifact_id,
                        "mimeType": content_type,
                        "path": str(path),
                        "sizeBytes": len(data),
                    }
                )
            elif content_type in ALLOWED_TEXT_TYPES or content_type.startswith("text/"):
                text = data[:MAX_TEXT_BYTES].decode("utf-8", errors="replace")
                extracted_text.append(
                    f"\n--- BEGIN UNTRUSTED ARTIFACT {artifact_id} ({content_type}) ---\n"
                    f"{text}\n--- END UNTRUSTED ARTIFACT {artifact_id} ---"
                )
            else:
                artifact["status"] = "unsupported"
                artifact["error"] = f"Unsupported artifact type: {content_type}"
            artifacts.append(artifact)
        except Exception as exc:
            artifacts.append(
                {
                    "id": artifact_id,
                    "sourceUrl": url,
                    "finalUrl": None,
                    "requirementKeys": requirement_keys,
                    "mimeType": None,
                    "sizeBytes": 0,
                    "sha256": None,
                    "status": "unreadable",
                    "error": str(exc)[:500],
                }
            )

    if len(url_requirements) > MAX_URLS:
        artifacts.append(
            {
                "id": "artifact-limit",
                "sourceUrl": None,
                "finalUrl": None,
                "requirementKeys": [],
                "mimeType": None,
                "sizeBytes": 0,
                "sha256": None,
                "status": "unreadable",
                "error": f"Submission contains more than {MAX_URLS} artifact URLs",
            }
        )

    stable_artifacts = [
        {key: value for key, value in artifact.items() if key != "finalUrl"}
        for artifact in artifacts
    ]
    canonical = json.dumps(stable_artifacts, sort_keys=True, separators=(",", ":"))
    manifest = {
        "schemaVersion": 1,
        "artifacts": artifacts,
        "totalBytes": consumed,
        "manifestHash": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    return manifest, media, "\n".join(extracted_text)[:MAX_EXTRACTED_TEXT_CHARS]


def _collect_urls(request: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    found: Dict[str, set[str]] = {}
    submission = request.get("submission") or {}
    sources = [(submission, "")]
    approved_architecture = request.get("approvedArchitecture")
    if isinstance(approved_architecture, dict):
        sources.append((approved_architecture, "approved_architecture:"))
    for source, key_prefix in sources:
        evidence = (source.get("content") or {}).get("requirementEvidence") or {}
        if not isinstance(evidence, dict):
            evidence = {}
        for key, item in evidence.items():
            if not isinstance(item, dict):
                continue
            if item.get("disposition") == "not_applicable":
                continue
            for url in item.get("urls") or []:
                if isinstance(url, str) and url.strip():
                    found.setdefault(url.strip(), set()).add(f"{key_prefix}{key}")
        for url in _nested_urls(source.get("fileUrls")):
            found.setdefault(url, set()).add(
                f"{key_prefix}unmapped" if key_prefix else "unmapped"
            )
    return [(url, sorted(keys)) for url, keys in found.items()]


def _nested_urls(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip().startswith(("https://", "http://")):
            yield value.strip()
    elif isinstance(value, list):
        for item in value:
            yield from _nested_urls(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nested_urls(item)


def _inspect_figma(
    source_url: str,
    artifact_id: str,
    requirement_keys: List[str],
    workspace: Path,
    previous_artifacts: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    token = os.getenv("FIGMA_ACCESS_TOKEN")
    if not token or token == "change-me":
        raise ArtifactInspectionError(
            "Figma artifact requires FIGMA_ACCESS_TOKEN with file_content:read scope"
        )
    key = _figma_key(source_url)
    if not key:
        raise ArtifactInspectionError("Invalid Figma file URL")
    headers = {"X-Figma-Token": token, "Accept": "application/json"}
    file_url = f"https://api.figma.com/v1/files/{urllib.parse.quote(key)}?depth=3"
    raw, content_type, _ = _download_fixed(file_url, MAX_FILE_BYTES, headers)
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ArtifactInspectionError("Figma returned invalid JSON") from exc

    version = str(document.get("version") or "")
    previous_root = next(
        (
            item
            for item in previous_artifacts
            if isinstance(item, dict)
            and item.get("id") == artifact_id
            and item.get("sourceUrl") == source_url
            and item.get("status") == "inspected"
        ),
        None,
    )
    if previous_root and version and previous_root.get("version") == version:
        previous_snapshot = [
            dict(item)
            for item in previous_artifacts
            if isinstance(item, dict)
            and (
                item.get("id") == artifact_id
                or item.get("parentArtifactId") == artifact_id
            )
        ]
        return (
            previous_snapshot,
            [],
            f"\nFigma artifact {artifact_id} is unchanged at version {version}; "
            "the previous immutable snapshot and verdict are reusable.",
        )
    structure = _figma_structure(document)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    root = _artifact_record(
        artifact_id,
        source_url,
        source_url,
        requirement_keys,
        "application/vnd.figma+json",
        canonical,
    )
    root["version"] = version or None
    root["metadata"] = structure
    artifacts = [root]
    media: List[Dict[str, Any]] = []

    frame_ids = [item["id"] for item in structure["frames"][:MAX_FIGMA_FRAMES]]
    if frame_ids:
        query = urllib.parse.urlencode(
            {"ids": ",".join(frame_ids), "format": "png", "scale": "1"}
        )
        image_api = f"https://api.figma.com/v1/images/{urllib.parse.quote(key)}?{query}"
        image_raw, _, _ = _download_fixed(image_api, 2 * 1024 * 1024, headers)
        image_map = (json.loads(image_raw).get("images") or {})
        for frame_index, frame in enumerate(structure["frames"][:MAX_FIGMA_FRAMES]):
            rendered_url = image_map.get(frame["id"])
            if not rendered_url:
                continue
            data, image_type, final_url = _download(rendered_url, 12 * 1024 * 1024)
            image_type = _normalize_content_type(image_type, final_url)
            if image_type not in {"image/png", "image/jpeg", "image/webp"}:
                continue
            frame_id = f"{artifact_id}-frame-{frame_index + 1}"
            path = workspace / f"{frame_id}{_extension(image_type, final_url)}"
            path.write_bytes(data)
            frame_artifact = _artifact_record(
                frame_id,
                source_url,
                final_url,
                requirement_keys,
                image_type,
                data,
            )
            frame_artifact["location"] = f"Figma frame {frame['id']} ({frame['name']})"
            frame_artifact["parentArtifactId"] = artifact_id
            artifacts.append(frame_artifact)
            media.append(
                {
                    "artifactId": frame_id,
                    "mimeType": image_type,
                    "path": str(path),
                    "sizeBytes": len(data),
                }
            )

    summary = json.dumps(
        {
            "artifactId": artifact_id,
            "figmaFileName": document.get("name"),
            "version": version,
            **structure,
        },
        indent=2,
    )
    return artifacts, media, f"\nFigma structural snapshot:\n{summary}"


def _figma_structure(document: Dict[str, Any]) -> Dict[str, Any]:
    pages: List[Dict[str, Any]] = []
    frames: List[Dict[str, Any]] = []
    type_counts: Dict[str, int] = {}
    total_frames = 0

    def visit(node: Any, page_name: str | None = None) -> None:
        nonlocal total_frames
        if not isinstance(node, dict):
            return
        node_type = str(node.get("type") or "UNKNOWN")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
        current_page = str(node.get("name") or "") if node_type == "CANVAS" else page_name
        if node_type == "CANVAS" and len(pages) < MAX_FIGMA_PAGES:
            pages.append(
                {"id": str(node.get("id"))[:200], "name": current_page[:200]}
            )
        if node_type in {"FRAME", "COMPONENT", "COMPONENT_SET", "SECTION"}:
            total_frames += 1
            if len(frames) < MAX_FIGMA_STRUCTURE_FRAMES:
                frames.append(
                    {
                        "id": str(node.get("id"))[:200],
                        "name": str(node.get("name") or "Unnamed")[:200],
                        "type": node_type,
                        "page": (current_page or "")[:200],
                    }
                )
        for child in node.get("children") or []:
            visit(child, current_page)

    visit(document.get("document"))
    return {
        "pages": pages,
        "frames": frames,
        "frameCount": total_frames,
        "structureTruncated": total_frames > len(frames),
        "nodeTypeCounts": type_counts,
        "componentCount": len(document.get("components") or {}),
        "componentSetCount": len(document.get("componentSets") or {}),
        "styleCount": len(document.get("styles") or {}),
    }


def _artifact_record(
    artifact_id: str,
    source_url: str,
    final_url: str,
    requirement_keys: List[str],
    content_type: str,
    data: bytes,
) -> Dict[str, Any]:
    return {
        "id": artifact_id,
        "sourceUrl": source_url,
        "finalUrl": final_url,
        "requirementKeys": requirement_keys,
        "mimeType": content_type,
        "sizeBytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "status": "inspected",
        "error": None,
    }


def _figma_key(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    if (parsed.hostname or "").lower() not in {"figma.com", "www.figma.com"}:
        return None
    match = FIGMA_URL_RE.match(parsed.path)
    return match.group(1) if match else None


def _download(
    url: str, limit: int, headers: Dict[str, str] | None = None
) -> Tuple[bytes, str, str]:
    current = url
    for _ in range(4):
        _validate_public_https_url(current)
        request = urllib.request.Request(
            current,
            headers={"User-Agent": "NexusArtifactInspector/1.0", **(headers or {})},
        )
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=20) as response:
                return _read_response(response, limit, current)
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                location = exc.headers.get("Location")
                if not location:
                    raise ArtifactInspectionError("Artifact redirect has no location")
                current = urllib.parse.urljoin(current, location)
                continue
            raise ArtifactInspectionError(f"Artifact returned HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise ArtifactInspectionError(f"Could not retrieve artifact: {exc.reason}") from exc
    raise ArtifactInspectionError("Artifact redirected too many times")


def _download_fixed(
    url: str, limit: int, headers: Dict[str, str]
) -> Tuple[bytes, str, str]:
    hostname = (urllib.parse.urlparse(url).hostname or "").lower()
    if hostname != "api.figma.com":
        raise ArtifactInspectionError("Unexpected fixed API host")
    request = urllib.request.Request(
        url, headers={"User-Agent": "NexusArtifactInspector/1.0", **headers}
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return _read_response(response, limit, url)
    except urllib.error.HTTPError as exc:
        raise ArtifactInspectionError(f"Figma API returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ArtifactInspectionError(f"Could not reach Figma API: {exc.reason}") from exc


def _read_response(response, limit: int, final_url: str) -> Tuple[bytes, str, str]:
    declared = response.headers.get("Content-Length")
    if declared and int(declared) > limit:
        raise ArtifactInspectionError(f"Artifact exceeds {limit} byte limit")
    chunks: List[bytes] = []
    size = 0
    while True:
        chunk = response.read(min(64 * 1024, limit + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            raise ArtifactInspectionError(f"Artifact exceeds {limit} byte limit")
    return b"".join(chunks), response.headers.get_content_type(), response.geturl() or final_url


def _validate_public_https_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ArtifactInspectionError("Artifact URLs must be credential-free HTTPS URLs")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror as exc:
        raise ArtifactInspectionError("Artifact hostname could not be resolved") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ArtifactInspectionError("Artifact URL resolves to a non-public address")


def _normalize_content_type(content_type: str, url: str) -> str:
    content_type = (content_type or "").split(";", 1)[0].strip().lower()
    if content_type in {"", "application/octet-stream", "binary/octet-stream"}:
        guessed, _ = mimetypes.guess_type(urllib.parse.urlparse(url).path)
        return guessed or "application/octet-stream"
    return content_type


def _extension(content_type: str, url: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix[:12]
    return suffix if suffix else (mimetypes.guess_extension(content_type) or ".bin")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
