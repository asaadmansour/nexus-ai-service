import base64
import io
import unittest
import zipfile

from app.agents.requirements.document_extraction import (
    _decode_document,
    _extract_docx_text,
    _keep_only_missing_fields,
    _normalize_document_platforms,
)


class RequirementsDocumentExtractionTests(unittest.TestCase):
    def test_invalid_base64_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid base64"):
            _decode_document("not-base64!")

    def test_docx_text_is_extracted_without_running_document_instructions(self):
        xml = b"""<?xml version='1.0' encoding='UTF-8'?>
        <w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>
          <w:body><w:p><w:r><w:t>Five-page clinic website</w:t></w:r></w:p></w:body>
        </w:document>"""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", xml)

        encoded = base64.b64encode(buffer.getvalue()).decode()
        self.assertTrue(_decode_document(encoded))
        self.assertEqual(_extract_docx_text(buffer.getvalue()), "Five-page clinic website")

    def test_existing_answers_are_not_overwritten_by_document(self):
        extracted, warnings = _keep_only_missing_fields(
            {
                "mainGoal": "Replace the existing goal",
                "integrations": "none",
            },
            {"mainGoal": "Collect clinic bookings"},
        )

        self.assertEqual(extracted, {"integrations": "none"})
        self.assertEqual(len(warnings), 1)
        self.assertIn("existing mainGoal", warnings[0])

    def test_mobile_website_document_does_not_create_native_app_scope(self):
        self.assertEqual(
            _normalize_document_platforms(
                {
                    "solutionType": "responsive mobile website",
                    "platforms": ["website", "mobile app"],
                }
            )["platforms"],
            ["website"],
        )


if __name__ == "__main__":
    unittest.main()
