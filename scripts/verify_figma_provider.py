"""Live Figma provider smoke test for CI and production diagnostics."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from app.agents.planning_artifacts import inspect_artifacts


def main() -> int:
    url = os.getenv("FIGMA_SMOKE_FILE_URL", "").strip()
    token = os.getenv("FIGMA_ACCESS_TOKEN", "").strip()
    if not token or token == "change-me" or not url:
        print(
            "Figma live smoke requires FIGMA_ACCESS_TOKEN and FIGMA_SMOKE_FILE_URL",
            file=sys.stderr,
        )
        return 2
    request = {
        "submission": {
            "submissionId": "figma-provider-smoke",
            "submissionVersion": 1,
            "submissionType": "ui_ux",
            "content": {
                "requirementEvidence": {
                    "prototype": {"summary": "Provider smoke", "urls": [url]}
                }
            },
            "fileUrls": {},
        }
    }
    with tempfile.TemporaryDirectory(prefix="nexus-figma-smoke-") as directory:
        manifest, media, extracted = inspect_artifacts(request, Path(directory))
    inspected = [
        item for item in manifest["artifacts"] if item.get("status") == "inspected"
    ]
    root = next(
        (
            item
            for item in inspected
            if item.get("mimeType") == "application/vnd.figma+json"
        ),
        None,
    )
    if not root or not root.get("sha256") or not root.get("version"):
        print(json.dumps(manifest, indent=2), file=sys.stderr)
        return 1
    if not media:
        print("Figma structure loaded but no frame render was retrieved", file=sys.stderr)
        return 1
    if "Figma structural snapshot" not in extracted:
        print("Figma structural snapshot was not extracted", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "version": root["version"],
                "manifestHash": manifest["manifestHash"],
                "renderedFrames": len(media),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
