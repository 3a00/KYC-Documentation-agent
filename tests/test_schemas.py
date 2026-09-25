"""Tests for Pydantic document extraction schemas."""

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    BusinessLicenseSchema,
    ExtractedField,
    NationalIDSchema,
    TaxCardSchema,
    normalize_arabic_numerals,
)


def test_normalize_arabic_numerals():
    """Verify Arabic-Indic digits are correctly converted to Western digits."""
    assert normalize_arabic_numerals("١٩٩٥١٢٣٤٥٦٧٨") == "199512345678"
    assert normalize_arabic_numerals(" 123-456 ") == "123456"
    assert normalize_arabic_numerals("١٢٣-٤٥٦ ٧٨٩") == "123456789"


def test_extracted_field_confidence_required():
    """ExtractedField must fail closed if confidence is omitted."""
    with pytest.raises(ValidationError):
        ExtractedField[str](value="Test", obscured=False)  # type: ignore[call-arg]


def test_extracted_field_confidence_bounds():
    """ExtractedField confidence must be clamped within [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        ExtractedField[str](value="Test", confidence=1.5, obscured=False)
    with pytest.raises(ValidationError):
        ExtractedField[str](value="Test", confidence=-0.1, obscured=False)

    field = ExtractedField[str](value="Test", confidence=0.85, obscured=False)
    assert field.confidence == 0.85


def test_zero_hallucination_obscured_with_value_fails():
    """ExtractedField must reject non-null values when obscured=True."""
    with pytest.raises(ValidationError, match="Zero-hallucination violation"):
        ExtractedField[str](value="Hallucination", confidence=0.5, obscured=True)


def test_zero_hallucination_null_with_unobscured_fails():
    """ExtractedField must reject null values when obscured=False."""
    with pytest.raises(ValidationError, match="Inconsistent field state"):
        ExtractedField[str](value=None, confidence=0.0, obscured=False)


def test_zero_hallucination_valid_obscured_field():
    """ExtractedField allows value=None when obscured=True."""
    field = ExtractedField[str](value=None, confidence=0.0, obscured=True)
    assert field.value is None
    assert field.obscured is True


def test_national_id_schema_valid():
    """Valid Iraqi National ID schema parses correctly."""
    schema = NationalIDSchema(
        national_id=ExtractedField(value="199512345678", confidence=0.98),
        full_name=ExtractedField(value="علي محمد حسن", confidence=0.95),
        mother_name=ExtractedField(value="فاطمة جاسم", confidence=0.88),
        expiry_date=ExtractedField(value="2029/05/12", confidence=0.92),
        issue_date=ExtractedField(value="2019/05/12", confidence=0.90),
        province=ExtractedField(value="بغداد", confidence=0.94),
    )
    assert schema.national_id.value == "199512345678"
    assert len(schema.get_tier1_fields()) == 3
    assert len(schema.get_tier2_fields()) == 3


def test_national_id_schema_arabic_indic_normalization():
    """National ID accepts 12-digit Arabic-Indic numerals."""
    schema = NationalIDSchema(
        national_id=ExtractedField(value="١٩٩٥١٢٣٤٥٦٧٨", confidence=0.98),
        full_name=ExtractedField(value="حسين علي صادق", confidence=0.95),
        mother_name=ExtractedField(value="زينب كريم", confidence=0.90),
        expiry_date=ExtractedField(value="2030/01/01", confidence=0.90),
        issue_date=ExtractedField(value="2020/01/01", confidence=0.90),
        province=ExtractedField(value="البصرة", confidence=0.90),
    )
    assert schema.national_id.value == "١٩٩٥١٢٣٤٥٦٧٨"


def test_national_id_schema_invalid_digit_count():
    """National ID fails validation if length is not exactly 12 digits."""
    with pytest.raises(ValidationError, match="12 numeric digits"):
        NationalIDSchema(
            national_id=ExtractedField(value="12345", confidence=0.90),
            full_name=ExtractedField(value="حسن جاسم", confidence=0.90),
            mother_name=ExtractedField(value="مريم", confidence=0.90),
            expiry_date=ExtractedField(value="2030/01/01", confidence=0.90),
            issue_date=ExtractedField(value="2020/01/01", confidence=0.90),
            province=ExtractedField(value="بغداد", confidence=0.90),
        )


def test_national_id_schema_obscured_skips_format_check():
    """Obscured National ID passes format validation without error."""
    schema = NationalIDSchema(
        national_id=ExtractedField(value=None, confidence=0.0, obscured=True),
        full_name=ExtractedField(value="حسن جاسم", confidence=0.90),
        mother_name=ExtractedField(value="مريم", confidence=0.90),
        expiry_date=ExtractedField(value="2030/01/01", confidence=0.90),
        issue_date=ExtractedField(value="2020/01/01", confidence=0.90),
        province=ExtractedField(value="بغداد", confidence=0.90),
    )
    assert schema.national_id.value is None
    assert schema.national_id.obscured is True


def test_business_license_schema_and_tiers():
    """Business license schema maps tier 1 and tier 2 fields accurately."""
    schema = BusinessLicenseSchema(
        business_name=ExtractedField(value="شركة النور للتجارة العامة", confidence=0.95),
        full_name=ExtractedField(value="أحمد كريم حميد", confidence=0.92),
        license_number=ExtractedField(value="BL-2023-8871", confidence=0.90),
        issue_date=ExtractedField(value="2023/04/10", confidence=0.85),
        business_activity=ExtractedField(value="تجارة عامة واستيراد", confidence=0.80),
        province=ExtractedField(value="بغداد", confidence=0.88),
    )
    tier1_fields = schema.get_tier1_fields()
    tier2_fields = schema.get_tier2_fields()
    assert "business_name" in tier1_fields
    assert "full_name" in tier1_fields
    assert "license_number" in tier1_fields
    assert "issue_date" in tier2_fields
    assert "business_activity" in tier2_fields
    assert "province" in tier2_fields


def test_tax_card_schema_and_tiers():
    """Tax card schema maps tier 1 and tier 2 fields accurately."""
    schema = TaxCardSchema(
        tax_id=ExtractedField(value="9001234567", confidence=0.94),
        taxpayer_name=ExtractedField(value="علي محمد حسن", confidence=0.91),
        fiscal_year=ExtractedField(value="2023", confidence=0.89),
        issue_date=ExtractedField(value="2023/01/15", confidence=0.85),
    )
    tier1_fields = schema.get_tier1_fields()
    tier2_fields = schema.get_tier2_fields()
    assert "tax_id" in tier1_fields
    assert "taxpayer_name" in tier1_fields
    assert "fiscal_year" in tier2_fields
    assert "issue_date" in tier2_fields
