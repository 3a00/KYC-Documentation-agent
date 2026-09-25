"""Pydantic extraction schemas enforcing calibrated confidence and zero-hallucination.

Defines schemas for Iraqi Unified National Card, Business License, and Tax Card.
Enforces that obscured fields must strictly have null values, preventing hallucinated
content from being accepted into the KYC verification pipeline.
"""

from typing import Generic, TypeVar
from pydantic import BaseModel, Field, field_validator, model_validator

T = TypeVar("T")

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"


def normalize_arabic_numerals(text: str) -> str:
    """Normalize Arabic-Indic numerals (٠-٩) and clean whitespace and hyphens."""
    normalized = text.strip().replace(" ", "").replace("-", "")
    for index, digit in enumerate(ARABIC_INDIC_DIGITS):
        normalized = normalized.replace(digit, str(index))
    return normalized


class ExtractedField(BaseModel, Generic[T]):
    """Generic wrapper for extracted identity document attributes.

    Confidence is strictly required with no default to fail closed on uncalibrated calls.
    """

    value: T | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    obscured: bool = False
    bbox: list[int] | tuple[int, int, int, int] | None = None

    @model_validator(mode="after")
    def validate_zero_hallucination(self) -> "ExtractedField[T]":
        """Strictly enforce biconditional zero-hallucination semantics.

        An obscured field must have value=None.
        A field with value=None must be marked as obscured=True.
        """
        if self.obscured and self.value is not None:
            raise ValueError(
                "Zero-hallucination violation: Field marked as obscured must have value=None"
            )
        if self.value is None and not self.obscured:
            raise ValueError(
                "Inconsistent field state: Field with value=None must be marked as obscured=True"
            )
        return self


class NationalIDSchema(BaseModel):
    """Extraction schema for Iraqi Unified National Card (البطاقة الوطنية الموحدة)."""

    national_id: ExtractedField[str]
    full_name: ExtractedField[str]
    mother_name: ExtractedField[str]
    expiry_date: ExtractedField[str]
    issue_date: ExtractedField[str]
    province: ExtractedField[str]

    @field_validator("national_id")
    @classmethod
    def validate_national_id_format(cls, field: ExtractedField[str]) -> ExtractedField[str]:
        """Enforce deterministic 12-digit numeric constraint for Iraqi National ID."""
        if not field.obscured:
            assert field.value is not None
            clean_digits = normalize_arabic_numerals(field.value)
            if len(clean_digits) != 12 or not clean_digits.isdigit():
                raise ValueError(
                    f"Iraqi National ID must be exactly 12 numeric digits, got: '{field.value}'"
                )
        return field

    def get_tier1_fields(self) -> dict[str, ExtractedField[str]]:
        """Return critical identity attributes requiring >= 0.85 calibrated confidence."""
        return {
            "national_id": self.national_id,
            "full_name": self.full_name,
            "expiry_date": self.expiry_date,
        }

    def get_tier2_fields(self) -> dict[str, ExtractedField[str]]:
        """Return contextual verification attributes requiring >= 0.70 confidence."""
        return {
            "mother_name": self.mother_name,
            "issue_date": self.issue_date,
            "province": self.province,
        }


class BusinessLicenseSchema(BaseModel):
    """Extraction schema for Iraqi Business License or Commercial Registry (إجازة ممارسة مهنة)."""

    business_name: ExtractedField[str]
    full_name: ExtractedField[str]
    license_number: ExtractedField[str]
    issue_date: ExtractedField[str]
    business_activity: ExtractedField[str] | None = None
    province: ExtractedField[str] | None = None

    def get_tier1_fields(self) -> dict[str, ExtractedField[str]]:
        """Return critical business attributes requiring >= 0.85 calibrated confidence."""
        return {
            "business_name": self.business_name,
            "full_name": self.full_name,
            "license_number": self.license_number,
        }

    def get_tier2_fields(self) -> dict[str, ExtractedField[str]]:
        """Return contextual business attributes requiring >= 0.70 confidence."""
        tier2: dict[str, ExtractedField[str]] = {"issue_date": self.issue_date}
        if self.business_activity is not None:
            tier2["business_activity"] = self.business_activity
        if self.province is not None:
            tier2["province"] = self.province
        return tier2


class TaxCardSchema(BaseModel):
    """Extraction schema for Iraqi General Commission for Taxes Card (الهوية الضريبية)."""

    tax_id: ExtractedField[str]
    taxpayer_name: ExtractedField[str]
    fiscal_year: ExtractedField[str]
    issue_date: ExtractedField[str]

    def get_tier1_fields(self) -> dict[str, ExtractedField[str]]:
        """Return critical tax attributes requiring >= 0.85 calibrated confidence."""
        return {
            "tax_id": self.tax_id,
            "taxpayer_name": self.taxpayer_name,
        }

    def get_tier2_fields(self) -> dict[str, ExtractedField[str]]:
        """Return contextual tax attributes requiring >= 0.70 confidence."""
        return {
            "fiscal_year": self.fiscal_year,
            "issue_date": self.issue_date,
        }



