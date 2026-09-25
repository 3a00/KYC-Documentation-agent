"""Unit tests for synthetic Iraqi document and defect generator."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from src.data.synthetic_generator import (
    generate_fictional_identity,
    generate_synthetic_document,
    get_fonts,
    format_arabic,
    PROVINCES,
)


class TestSyntheticDocumentGenerator(unittest.TestCase):
    """Test suite for synthetic document generation and defect injection."""

    def test_fictional_identity_structure(self) -> None:
        """Verify fictional identity conforms to Iraqi standards and zero real customer rules."""
        identity = generate_fictional_identity(seed=123)

        # 4-part patronymic name chain
        name_parts = identity["full_name"].split()
        self.assertGreaterEqual(len(name_parts), 4)

        # 12-digit numeric national ID
        self.assertEqual(len(identity["national_id"]), 12)
        self.assertTrue(identity["national_id"].isdigit())

        # Province from curated list
        self.assertIn(identity["province"], PROVINCES)

        # Date format YYYY/MM/DD
        self.assertEqual(len(identity["issue_date"].split("/")), 3)
        self.assertEqual(len(identity["expiry_date"].split("/")), 3)

        # Deterministic seed test
        identity_repeat = generate_fictional_identity(seed=123)
        self.assertEqual(identity, identity_repeat)

    def test_format_arabic_reshaping(self) -> None:
        """Verify Arabic text reshaping and bidi reordering produces non-empty string."""
        sample_arabic = "علي محمد"
        formatted = format_arabic(sample_arabic)
        self.assertIsInstance(formatted, str)
        self.assertGreater(len(formatted), 0)

    def test_font_validation_loud_failure(self) -> None:
        """Verify RuntimeError is raised loudly when required fonts are missing."""
        with patch("src.data.synthetic_generator.FONT_REGULAR", "/nonexistent/font.ttf"):
            with self.assertRaises(RuntimeError) as context:
                get_fonts()
            self.assertIn("Required authentic Arabic font not found", str(context.exception))

    def test_unified_national_card_generation(self) -> None:
        """Verify National Card template rendering and bounding boxes."""
        result = generate_synthetic_document("unified_national_card", seed=42)
        self.assertEqual(result.document_type, "unified_national_card")
        self.assertIsInstance(result.clean_image, Image.Image)
        self.assertIsInstance(result.degraded_image, Image.Image)
        self.assertEqual(result.clean_image.size, (850, 540))

        # Check required fields
        expected_fields = {
            "full_name",
            "mother_name",
            "national_id",
            "province",
            "issue_date",
            "expiry_date",
        }
        self.assertTrue(expected_fields.issubset(set(result.fields.keys())))

        # Validate bounding box coordinates
        width, height = result.clean_image.size
        for field_name, field_obj in result.fields.items():
            xmin, ymin, xmax, ymax = field_obj.bbox
            self.assertLess(xmin, xmax, f"Field {field_name} xmin >= xmax")
            self.assertLess(ymin, ymax, f"Field {field_name} ymin >= ymax")
            self.assertGreaterEqual(xmin, 0)
            self.assertGreaterEqual(ymin, 0)
            self.assertLessEqual(xmax, width)
            self.assertLessEqual(ymax, height)

    def test_business_license_generation(self) -> None:
        """Verify Business License template rendering and bounding boxes."""
        result = generate_synthetic_document("business_license", seed=42)
        self.assertEqual(result.document_type, "business_license")
        self.assertEqual(result.clean_image.size, (750, 950))

        expected_fields = {
            "license_number",
            "business_name",
            "full_name",
            "business_activity",
            "issue_date",
            "province",
        }
        self.assertTrue(expected_fields.issubset(set(result.fields.keys())))

        for field_name, field_obj in result.fields.items():
            xmin, ymin, xmax, ymax = field_obj.bbox
            self.assertLess(xmin, xmax)
            self.assertLess(ymin, ymax)

    def test_tax_card_generation(self) -> None:
        """Verify Tax Card template rendering and bounding boxes."""
        result = generate_synthetic_document("tax_card", seed=42)
        self.assertEqual(result.document_type, "tax_card")
        self.assertEqual(result.clean_image.size, (800, 500))

        expected_fields = {
            "tax_id",
            "taxpayer_name",
            "fiscal_year",
            "issue_date",
        }
        self.assertTrue(expected_fields.issubset(set(result.fields.keys())))

        for field_name, field_obj in result.fields.items():
            xmin, ymin, xmax, ymax = field_obj.bbox
            self.assertLess(xmin, xmax)
            self.assertLess(ymin, ymax)

    def test_defect_injection_specular_glare(self) -> None:
        """Verify targeted specular glare injection modifies image and metadata."""
        result = generate_synthetic_document(
            "unified_national_card",
            defect_types=["glare"],
            target_field="national_id",
            seed=42,
        )
        self.assertEqual(len(result.defects_applied), 1)
        defect = result.defects_applied[0]
        self.assertEqual(defect["type"], "specular_glare")
        self.assertEqual(defect["target_field"], "national_id")
        self.assertEqual(defect["bbox"], list(result.fields["national_id"].bbox))

        # Check that degraded image differs from clean image
        clean_bytes = result.clean_image.tobytes()
        degraded_bytes = result.degraded_image.tobytes()
        self.assertNotEqual(clean_bytes, degraded_bytes)

    def test_defect_injection_gaussian_blur(self) -> None:
        """Verify Gaussian blur defect modifies image and records metadata."""
        result = generate_synthetic_document(
            "unified_national_card",
            defect_types=["blur"],
            seed=42,
        )
        self.assertEqual(len(result.defects_applied), 1)
        self.assertEqual(result.defects_applied[0]["type"], "gaussian_blur")
        self.assertNotEqual(result.clean_image.tobytes(), result.degraded_image.tobytes())

    def test_defect_injection_perspective_tilt(self) -> None:
        """Verify perspective tilt defect modifies image and records metadata."""
        result = generate_synthetic_document(
            "unified_national_card",
            defect_types=["tilt"],
            seed=42,
        )
        self.assertEqual(len(result.defects_applied), 1)
        self.assertEqual(result.defects_applied[0]["type"], "perspective_tilt")
        self.assertNotEqual(result.clean_image.tobytes(), result.degraded_image.tobytes())

    def test_chained_defects(self) -> None:
        """Verify compound defects (glare + tilt) can be applied simultaneously."""
        result = generate_synthetic_document(
            "unified_national_card",
            defect_types=["glare", "tilt"],
            seed=42,
        )
        self.assertEqual(len(result.defects_applied), 2)
        types = [defect_item["type"] for defect_item in result.defects_applied]
        self.assertEqual(types, ["specular_glare", "perspective_tilt"])

    def test_to_metadata_json_serializable(self) -> None:
        """Verify metadata serialization produces valid JSON structure."""
        result = generate_synthetic_document(
            "unified_national_card",
            defect_types=["glare"],
            seed=42,
        )
        metadata = result.to_metadata()
        serialized = json.dumps(metadata)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized["document_type"], "unified_national_card")
        self.assertIn("national_id", deserialized["fields"])
        self.assertEqual(len(deserialized["defects_applied"]), 1)

    def test_save_paired_outputs(self) -> None:
        """Verify saving clean image, degraded image, and metadata to disk."""
        result = generate_synthetic_document(
            "unified_national_card",
            defect_types=["glare"],
            seed=42,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            clean_path, degraded_path, meta_path = result.save(
                temporary_directory, prefix="test_card"
            )
            self.assertTrue(clean_path.exists())
            self.assertTrue(degraded_path.exists())
            self.assertTrue(meta_path.exists())

            # Verify saved metadata content
            with open(meta_path, "r", encoding="utf-8") as metadata_file:
                loaded_metadata = json.load(metadata_file)
            self.assertEqual(loaded_metadata["document_type"], "unified_national_card")

    def test_invalid_document_type(self) -> None:
        """Verify ValueError on unsupported document type."""
        with self.assertRaises(ValueError):
            generate_synthetic_document("passport")

    def test_invalid_defect_type(self) -> None:
        """Verify ValueError on unsupported defect type."""
        with self.assertRaises(ValueError):
            generate_synthetic_document("unified_national_card", defect_types=["fire"])


if __name__ == "__main__":
    unittest.main()
