"""Synthetic Iraqi document and defect generator.

Generates paired synthetic identity documents (Unified National Card,
Business License, Tax Card) with authentic Iraqi layouts, Arabic typography,
exact pixel bounding boxes, and localized optical defects (glare, blur, tilt).
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

FONT_REGULAR = "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf"

# Fictional combinatorial Iraqi identity data (Zero real customer data)
GIVEN_NAMES = [
    "علي", "محمد", "حسين", "أحمد", "حسن", "كرار",
    "سجاد", "عمر", "يوسف", "مصطفى", "حيدر", "عباس", "زيد"
]
FATHER_NAMES = [
    "كاظم", "جاسم", "كريم", "صادق", "إبراهيم", "خضير",
    "شاكر", "مهدي", "طارق", "حميد", "رزاق", "سلمان"
]
GRANDFATHER_NAMES = [
    "عبد الله", "محمود", "عزيز", "جبار", "فاضل",
    "رشيد", "باقر", "عدنان", "جليل", "حمزة"
]
SURNAMES = [
    "الزبيدي", "الجبوري", "الخفاجي", "الساعدي", "الشمري",
    "التميمي", "العبيدي", "المالكي", "الدليمي", "البياتي"
]
MOTHER_NAMES = [
    "فاطمة حسن", "زينب كريم", "مريم جاسم", "هدى كاظم",
    "زهراء علي", "نور صادق", "إيمان حميد"
]
PROVINCES = [
    "بغداد", "البصرة", "نينوى", "أربيل", "النجف", "كربلاء",
    "بابل", "ذي قار", "كركوك", "الأنبار", "ديالى", "واسط"
]
BUSINESS_TYPES = [
    "تجارة المواد الغذائية", "تجارة الأجهزة الكهربائية",
    "مقاولات عامة وتجهيزات", "خدمات حاسوب واتصالات"
]


@dataclass
class DocumentField:
    """Represents an extracted or ground-truth field with exact bounding box."""

    name: str
    value: str
    bbox: tuple[int, int, int, int]


@dataclass
class SyntheticDocumentResult:
    """Encapsulates paired clean and degraded document images and ground truth."""

    document_type: str
    clean_image: Image.Image
    degraded_image: Image.Image
    fields: dict[str, DocumentField]
    defects_applied: list[dict[str, Any]]

    def to_metadata(self) -> dict[str, Any]:
        """Serialize document metadata and exact field bounding boxes."""
        return {
            "document_type": self.document_type,
            "fields": {
                field_name: {
                    "value": field_obj.value,
                    "bbox": list(field_obj.bbox),
                }
                for field_name, field_obj in self.fields.items()
            },
            "defects_applied": self.defects_applied,
        }

    def save(
        self,
        output_dir: Path | str,
        prefix: str = "document",
    ) -> tuple[Path, Path, Path]:
        """Save clean image, degraded image, and ground-truth metadata to disk."""
        target_directory = Path(output_dir)
        target_directory.mkdir(parents=True, exist_ok=True)
        clean_path = target_directory / f"{prefix}_clean.png"
        degraded_path = target_directory / f"{prefix}_degraded.png"
        metadata_path = target_directory / f"{prefix}_meta.json"

        self.clean_image.save(clean_path)
        self.degraded_image.save(degraded_path)
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(self.to_metadata(), metadata_file, ensure_ascii=False, indent=2)

        return clean_path, degraded_path, metadata_path


def format_arabic(text: str) -> str:
    """Reshape and reorder Arabic text for correct RTL visual rendering."""
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


def get_fonts(base_size: int = 18) -> dict[str, ImageFont.FreeTypeFont]:
    """Load authentic Noto typography at varied weights and sizes.

    Fails loudly if required Arabic fonts are missing from the system.
    """
    for font_path in (FONT_REGULAR, FONT_BOLD):
        if not Path(font_path).exists():
            raise RuntimeError(
                f"Required authentic Arabic font not found at '{font_path}'. "
                "Install fonts-noto-core on Debian/Ubuntu or ensure Noto fonts are in /usr/share/fonts."
            )
    return {
        "title": ImageFont.truetype(FONT_BOLD, int(base_size * 1.3)),
        "header": ImageFont.truetype(FONT_BOLD, base_size),
        "label": ImageFont.truetype(FONT_REGULAR, int(base_size * 0.85)),
        "value": ImageFont.truetype(FONT_BOLD, base_size),
        "value_small": ImageFont.truetype(FONT_REGULAR, int(base_size * 0.9)),
    }


def draw_arabic_field(
    draw: ImageDraw.ImageDraw,
    label: str,
    value: str,
    position: tuple[int, int],
    fonts: dict[str, ImageFont.FreeTypeFont],
    anchor: str = "ra",
) -> tuple[int, int, int, int]:
    """Draw Arabic field with RTL layout and record exact value bounding box."""
    x_pos, y_pos = position
    formatted_label = format_arabic(label)
    draw.text(
        (x_pos, y_pos),
        formatted_label,
        font=fonts["label"],
        fill=(90, 100, 110),
        anchor=anchor,
    )

    value_y = y_pos + 24
    formatted_value = format_arabic(value)
    bbox = draw.textbbox(
        (x_pos, value_y),
        formatted_value,
        font=fonts["value"],
        anchor=anchor,
    )
    draw.text(
        (x_pos, value_y),
        formatted_value,
        font=fonts["value"],
        fill=(20, 30, 45),
        anchor=anchor,
    )
    return bbox


def generate_fictional_identity(seed: int | None = None) -> dict[str, str]:
    """Synthesize fictional Iraqi identity attributes strictly adhering to zero real customer rules."""
    # yagni: in-memory combinatorial lists; external generator if test set exceeds 10k rows
    rng = random.Random(seed)
    given = rng.choice(GIVEN_NAMES)
    father = rng.choice(FATHER_NAMES)
    grandfather = rng.choice(GRANDFATHER_NAMES)
    surname = rng.choice(SURNAMES)
    full_name = f"{given} {father} {grandfather} {surname}"

    birth_year = rng.randint(1975, 2003)
    national_id = f"{birth_year}{rng.randint(10000000, 99999999)}"  # Exactly 12 digits

    issue_year = rng.randint(2018, 2023)
    issue_month = rng.randint(1, 12)
    issue_day = rng.randint(1, 28)
    issue_date = f"{issue_year}/{issue_month:02d}/{issue_day:02d}"
    expiry_date = f"{issue_year + 10}/{issue_month:02d}/{issue_day:02d}"

    return {
        "full_name": full_name,
        "father_name": father,
        "mother_name": rng.choice(MOTHER_NAMES),
        "national_id": national_id,
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "province": rng.choice(PROVINCES),
        "business_name": f"شركة {rng.choice(GIVEN_NAMES)} {rng.choice(SURNAMES)} للتجارة المحدودة",
        "business_activity": rng.choice(BUSINESS_TYPES),
        "license_number": f"BL-{issue_year}-{rng.randint(1000, 9999)}",
        "tax_id": f"TIN-{rng.randint(1000000, 9999999)}",
        "fiscal_year": str(issue_year),
    }


def render_national_card(
    identity: dict[str, str],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> tuple[Image.Image, dict[str, DocumentField]]:
    """Render authentic synthetic front-face Iraqi Unified National Card."""
    width, height = 850, 540
    card = Image.new("RGB", (width, height), (242, 246, 250))
    draw = ImageDraw.Draw(card)

    # Outer border and header banner
    draw.rounded_rectangle(
        [(15, 15), (width - 15, height - 15)],
        radius=18,
        outline=(170, 185, 205),
        width=3,
    )
    draw.rectangle([(20, 20), (width - 20, 85)], fill=(32, 60, 100))

    header_text = format_arabic("جمهورية العراق - وزارة الداخلية - البطاقة الوطنية الموحدة")
    draw.text(
        (width // 2, 52),
        header_text,
        font=fonts["title"],
        fill=(255, 255, 255),
        anchor="mm",
    )

    # Photo box placeholder (Right-hand side per standard Iraqi National Card layout)
    photo_box = [(width - 210, 110), (width - 35, 340)]
    draw.rectangle(photo_box, fill=(225, 232, 242), outline=(150, 170, 195), width=2)
    draw.text(
        (width - 122, 225),
        format_arabic("صورة شخصية"),
        font=fonts["label"],
        fill=(120, 140, 160),
        anchor="mm",
    )

    fields: dict[str, DocumentField] = {}
    base_x = width - 240

    bbox = draw_arabic_field(draw, "الاسم الكامل", identity["full_name"], (base_x, 110), fonts)
    fields["full_name"] = DocumentField("full_name", identity["full_name"], bbox)

    bbox = draw_arabic_field(draw, "اسم الأم", identity["mother_name"], (base_x, 185), fonts)
    fields["mother_name"] = DocumentField("mother_name", identity["mother_name"], bbox)

    bbox = draw_arabic_field(draw, "رقم الهوية الوطنية", identity["national_id"], (base_x, 260), fonts)
    fields["national_id"] = DocumentField("national_id", identity["national_id"], bbox)

    # Bottom metadata row
    col3_x = base_x
    col2_x = base_x - 190
    col1_x = base_x - 380

    bbox = draw_arabic_field(draw, "المحافظة", identity["province"], (col3_x, 355), fonts)
    fields["province"] = DocumentField("province", identity["province"], bbox)

    bbox = draw_arabic_field(draw, "تاريخ الإصدار", identity["issue_date"], (col2_x, 355), fonts)
    fields["issue_date"] = DocumentField("issue_date", identity["issue_date"], bbox)

    bbox = draw_arabic_field(draw, "تاريخ النفاذ", identity["expiry_date"], (col1_x, 355), fonts)
    fields["expiry_date"] = DocumentField("expiry_date", identity["expiry_date"], bbox)

    return card, fields


def render_business_license(
    identity: dict[str, str],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> tuple[Image.Image, dict[str, DocumentField]]:
    """Render authentic synthetic Iraqi Business License / Commercial Registry document."""
    width, height = 750, 950
    cert = Image.new("RGB", (width, height), (252, 252, 246))
    draw = ImageDraw.Draw(cert)

    draw.rectangle([(25, 25), (width - 25, height - 25)], outline=(140, 120, 80), width=4)
    draw.rectangle([(35, 35), (width - 35, height - 35)], outline=(190, 170, 130), width=1)

    title_text = format_arabic("جمهورية العراق - وزارة التجارة")
    draw.text((width // 2, 80), title_text, font=fonts["header"], fill=(60, 50, 30), anchor="mm")
    sub_title = format_arabic("دائرة تسجيل الشركات - إجازة ممارسة مهنة وسجل تجاري")
    draw.text((width // 2, 120), sub_title, font=fonts["title"], fill=(120, 30, 30), anchor="mm")

    fields: dict[str, DocumentField] = {}
    center_x = width - 80

    bbox = draw_arabic_field(draw, "رقم السجل التجاري", identity["license_number"], (center_x, 190), fonts)
    fields["license_number"] = DocumentField("license_number", identity["license_number"], bbox)

    bbox = draw_arabic_field(draw, "اسم المنشأة التجارية", identity["business_name"], (center_x, 290), fonts)
    fields["business_name"] = DocumentField("business_name", identity["business_name"], bbox)

    bbox = draw_arabic_field(draw, "اسم التاجر / المدير المفوض", identity["full_name"], (center_x, 390), fonts)
    fields["full_name"] = DocumentField("full_name", identity["full_name"], bbox)

    bbox = draw_arabic_field(draw, "نوع النشاط التجاري", identity["business_activity"], (center_x, 490), fonts)
    fields["business_activity"] = DocumentField("business_activity", identity["business_activity"], bbox)

    bbox = draw_arabic_field(draw, "تاريخ الإصدار", identity["issue_date"], (center_x, 590), fonts)
    fields["issue_date"] = DocumentField("issue_date", identity["issue_date"], bbox)

    bbox = draw_arabic_field(draw, "المدينة / المحافظة", identity["province"], (center_x, 690), fonts)
    fields["province"] = DocumentField("province", identity["province"], bbox)

    # Official seal placeholder
    seal_box = [(90, 750), (220, 880)]
    draw.ellipse(seal_box, outline=(150, 40, 40), width=3)
    draw.text((155, 815), format_arabic("الختم الرسمي"), font=fonts["label"], fill=(150, 40, 40), anchor="mm")

    return cert, fields


def render_tax_card(
    identity: dict[str, str],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> tuple[Image.Image, dict[str, DocumentField]]:
    """Render authentic synthetic Iraqi General Commission for Taxes identification card."""
    width, height = 800, 500
    card = Image.new("RGB", (width, height), (248, 247, 242))
    draw = ImageDraw.Draw(card)

    draw.rounded_rectangle([(15, 15), (width - 15, height - 15)], radius=15, outline=(120, 140, 110), width=3)
    draw.rectangle([(20, 20), (width - 20, 80)], fill=(45, 95, 60))

    header_text = format_arabic("جمهورية العراق - الهيئة العامة للضرائب")
    draw.text((width // 2, 50), header_text, font=fonts["title"], fill=(255, 255, 255), anchor="mm")

    fields: dict[str, DocumentField] = {}
    base_x = width - 70

    bbox = draw_arabic_field(draw, "الرقم الضريبي للمكلف", identity["tax_id"], (base_x, 120), fonts)
    fields["tax_id"] = DocumentField("tax_id", identity["tax_id"], bbox)

    bbox = draw_arabic_field(draw, "اسم المكلف / التاجر", identity["full_name"], (base_x, 210), fonts)
    fields["taxpayer_name"] = DocumentField("taxpayer_name", identity["full_name"], bbox)

    bbox = draw_arabic_field(draw, "السنة المالية", identity["fiscal_year"], (base_x, 300), fonts)
    fields["fiscal_year"] = DocumentField("fiscal_year", identity["fiscal_year"], bbox)

    bbox = draw_arabic_field(draw, "تاريخ الإصدار", identity["issue_date"], (base_x - 350, 300), fonts)
    fields["issue_date"] = DocumentField("issue_date", identity["issue_date"], bbox)

    return card, fields


# Optical Defect Injection Operators


def inject_specular_glare(
    image: Image.Image,
    target_bbox: tuple[int, int, int, int],
    intensity: float = 0.85,
) -> Image.Image:
    """Inject realistic specular glare hotspot placed directly over target bounding box."""
    # yagni: radial 2D Gaussian glare; 3D ray-tracing shader if physical reflection calibration demands it
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    xmin, ymin, xmax, ymax = target_bbox
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    radius = max((xmax - xmin), (ymax - ymin), 50)

    y_grid, x_grid = np.ogrid[:height, :width]
    dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    sigma = radius / 1.7
    glare_mask = np.exp(-dist_sq / (2 * (sigma ** 2)))

    glare_layer = (glare_mask * 255 * intensity).clip(0, 255).astype(np.uint8)
    glare_bgr = cv2.merge([glare_layer, glare_layer, glare_layer])
    result_bgr = cv2.add(image_bgr, glare_bgr)
    return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))


def inject_gaussian_blur(image: Image.Image, kernel_size: int = 15) -> Image.Image:
    """Inject optical blur simulating out-of-focus smartphone camera."""
    # yagni: whole-image Gaussian blur; depth-of-field mesh when lens-tilt modeling is required
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred_bgr = cv2.GaussianBlur(image_bgr, (kernel_size, kernel_size), 0)
    return Image.fromarray(cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB))


def inject_perspective_tilt(image: Image.Image, max_tilt_ratio: float = 0.08) -> Image.Image:
    """Inject perspective tilt simulating handheld camera angle."""
    # yagni: 4-corner affine/projective warp; full 6-DOF camera pose rotation if 3D calibration needed
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]
    delta_x = width * max_tilt_ratio
    delta_y = height * max_tilt_ratio

    src_corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst_corners = np.float32([
        [delta_x * 0.5, delta_y * 0.7],
        [width - delta_x * 0.6, delta_y * 0.3],
        [width - delta_x * 0.4, height - delta_y * 0.8],
        [delta_x * 0.5, height - delta_y * 0.4],
    ])
    matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped_bgr = cv2.warpPerspective(image_bgr, matrix, (width, height), borderValue=(240, 240, 240))
    return Image.fromarray(cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB))


def generate_synthetic_document(
    document_type: str = "unified_national_card",
    defect_types: list[str] | str | None = None,
    target_field: str | None = None,
    seed: int | None = None,
) -> SyntheticDocumentResult:
    """Primary pipeline generator emitting paired clean image, degraded image, and exact field bboxes.

    Supports chaining multiple defects (e.g. ['glare', 'tilt']) to simulate realistic capture noise.
    """
    identity = generate_fictional_identity(seed=seed)
    fonts = get_fonts(base_size=20)

    if document_type == "unified_national_card":
        clean_img, fields = render_national_card(identity, fonts)
    elif document_type == "business_license":
        clean_img, fields = render_business_license(identity, fonts)
    elif document_type == "tax_card":
        clean_img, fields = render_tax_card(identity, fonts)
    else:
        raise ValueError(f"Unsupported document type: {document_type}")

    degraded_img = clean_img.copy()
    defects_applied: list[dict[str, Any]] = []

    if isinstance(defect_types, str):
        active_defects = [defect_types]
    elif isinstance(defect_types, list):
        active_defects = defect_types
    else:
        active_defects = []

    for defect in active_defects:
        if defect == "glare":
            field_to_target = target_field if target_field in fields else list(fields.keys())[0]
            target_bbox = fields[field_to_target].bbox
            degraded_img = inject_specular_glare(degraded_img, target_bbox)
            defects_applied.append({
                "type": "specular_glare",
                "target_field": field_to_target,
                "bbox": list(target_bbox),
            })
        elif defect == "blur":
            degraded_img = inject_gaussian_blur(degraded_img)
            defects_applied.append({"type": "gaussian_blur", "kernel_size": 15})
        elif defect == "tilt":
            degraded_img = inject_perspective_tilt(degraded_img)
            defects_applied.append({"type": "perspective_tilt", "max_tilt_ratio": 0.08})
        else:
            raise ValueError(f"Unsupported defect type: {defect}")

    return SyntheticDocumentResult(
        document_type=document_type,
        clean_image=clean_img,
        degraded_image=degraded_img,
        fields=fields,
        defects_applied=defects_applied,
    )


if __name__ == "__main__":
    # Self-verification testing all document types and every defect individually and combined
    test_defect_suites = [
        [],                     # Clean baseline
        ["glare"],              # Individual glare
        ["blur"],               # Individual blur
        ["tilt"],               # Individual tilt
        ["glare", "tilt"],      # Compound defect: glare + camera tilt
    ]

    for doc_type_name in ["unified_national_card", "business_license", "tax_card"]:
        for defect_list in test_defect_suites:
            document_pair = generate_synthetic_document(
                document_type=doc_type_name,
                defect_types=defect_list,
                seed=42,
            )
            assert document_pair.clean_image.size == document_pair.degraded_image.size
            assert len(document_pair.fields) >= 4
            for single_field_name, field_object in document_pair.fields.items():
                assert field_object.bbox[2] > field_object.bbox[0]
                assert field_object.bbox[3] > field_object.bbox[1]
            document_metadata = document_pair.to_metadata()
            assert document_metadata["document_type"] == doc_type_name
            assert len(document_metadata["defects_applied"]) == len(defect_list)

    print("All generator assertions (clean, individual, and compound defects) passed successfully!")
