import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.agents.planning_artifacts import _inspect_figma


class FigmaProviderContractTests(unittest.TestCase):
    def test_file_structure_and_rendered_frames_are_snapshotted(self):
        document = {
            "name": "Nexus smoke design",
            "version": "42",
            "document": {
                "id": "root",
                "name": "Document",
                "type": "DOCUMENT",
                "children": [
                    {
                        "id": "page:1",
                        "name": "Main",
                        "type": "CANVAS",
                        "children": [
                            {"id": "frame:1", "name": "Dashboard", "type": "FRAME"}
                        ],
                    }
                ],
            },
            "components": {},
            "componentSets": {},
            "styles": {},
        }
        images = {"images": {"frame:1": "https://cdn.example.com/frame.png"}}
        responses = [
            (json.dumps(document).encode(), "application/json", "https://api.figma.com/file"),
            (json.dumps(images).encode(), "application/json", "https://api.figma.com/images"),
        ]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"FIGMA_ACCESS_TOKEN": "real-token"}
        ), patch(
            "app.agents.planning_artifacts._download_fixed", side_effect=responses
        ), patch(
            "app.agents.planning_artifacts._download",
            return_value=(b"png-data", "image/png", "https://cdn.example.com/frame.png"),
        ):
            artifacts, media, extracted = _inspect_figma(
                "https://www.figma.com/design/file-key/Nexus",
                "artifact-1",
                ["prototype"],
                Path(directory),
                [],
            )

        self.assertEqual(artifacts[0]["version"], "42")
        self.assertEqual(artifacts[0]["metadata"]["frameCount"], 1)
        self.assertEqual(artifacts[1]["status"], "inspected")
        self.assertEqual(media[0]["artifactId"], artifacts[1]["id"])
        self.assertIn("Figma structural snapshot", extracted)


if __name__ == "__main__":
    unittest.main()
