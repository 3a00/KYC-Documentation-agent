"""OpenCV image quality diagnostic sensors and confidence calibration.

Computes Laplacian blur variance and HSV color-space specular glare ratios
for whole document images and localized bounding boxes. Emits structured
quality metrics without acting as hard reject gates.
"""

from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityMetrics:
    """Structured image quality diagnostic indicators."""

    blur_variance: float
    glare_ratio: float
    is_blurry: bool
    has_glare: bool
    clarity_factor: float


def to_cv2_bgr(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert PIL Image or numpy array to OpenCV BGR format."""
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f"Expected PIL.Image.Image or np.ndarray, got {type(image)}")


def crop_bounding_box(
    bgr_image: np.ndarray,
    bbox: tuple[int, int, int, int] | list[int] | None = None,
) -> np.ndarray:
    """Safely crop bounding box (xmin, ymin, xmax, ymax) from image."""
    if bbox is None:
        return bgr_image

    height, width = bgr_image.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    xmin_clamped = max(0, min(width, int(xmin)))
    xmax_clamped = max(0, min(width, int(xmax)))
    ymin_clamped = max(0, min(height, int(ymin)))
    ymax_clamped = max(0, min(height, int(ymax)))

    if xmax_clamped > xmin_clamped and ymax_clamped > ymin_clamped:
        return bgr_image[ymin_clamped:ymax_clamped, xmin_clamped:xmax_clamped]
    return bgr_image


def compute_blur_variance(
    image: Image.Image | np.ndarray,
    bbox: tuple[int, int, int, int] | list[int] | None = None,
) -> float:
    """Compute Laplacian variance measuring focus sharpness.

    Higher values indicate sharp edges; lower values indicate optical blur.
    """
    bgr = to_cv2_bgr(image)
    target = crop_bounding_box(bgr, bbox)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_glare_ratio(
    image: Image.Image | np.ndarray,
    bbox: tuple[int, int, int, int] | list[int] | None = None,
    v_threshold: int = 253,
    s_threshold: int = 35,
) -> float:
    """Compute ratio of pixels exhibiting specular glare in HSV space.

    Specular glare is characterized by saturated brightness (V >= v_threshold)
    and low saturation (S <= s_threshold).
    """
    bgr = to_cv2_bgr(image)
    target = crop_bounding_box(bgr, bbox)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    glare_mask = (v_channel >= v_threshold) & (s_channel <= s_threshold)
    total_pixels = target.shape[0] * target.shape[1]
    if total_pixels == 0:
        return 0.0
    return float(np.sum(glare_mask) / total_pixels)


def assess_quality(
    image: Image.Image | np.ndarray,
    bbox: tuple[int, int, int, int] | list[int] | None = None,
    blur_threshold: float = 100.0,
    glare_threshold: float = 0.50,
    v_threshold: int = 253,
    s_threshold: int = 35,
) -> QualityMetrics:
    """Evaluate image quality metrics for an image or localized bounding box.

    Returns structured metrics rather than acting as a hard reject gate.
    """
    blur_var = compute_blur_variance(image, bbox=bbox)
    glare_rat = compute_glare_ratio(
        image, bbox=bbox, v_threshold=v_threshold, s_threshold=s_threshold
    )

    # Local clarity factor scaled to [0.0, 1.0]
    blur_score = min(1.0, max(0.0, blur_var / 250.0))
    glare_penalty = max(0.0, 1.0 - (glare_rat * 1.5))
    clarity_factor = float(np.clip(blur_score * glare_penalty, 0.0, 1.0))

    return QualityMetrics(
        blur_variance=blur_var,
        glare_ratio=glare_rat,
        is_blurry=(blur_var < blur_threshold),
        has_glare=(glare_rat > glare_threshold),
        clarity_factor=clarity_factor,
    )


def assess_field_qualities(
    image: Image.Image | np.ndarray,
    field_bboxes: dict[str, tuple[int, int, int, int] | list[int]],
    blur_threshold: float = 100.0,
    glare_threshold: float = 0.50,
    v_threshold: int = 253,
    s_threshold: int = 35,
) -> dict[str, QualityMetrics]:
    """Assess localized quality metrics for a mapping of field names to bounding boxes."""
    return {
        field_name: assess_quality(
            image=image,
            bbox=bbox,
            blur_threshold=blur_threshold,
            glare_threshold=glare_threshold,
            v_threshold=v_threshold,
            s_threshold=s_threshold,
        )
        for field_name, bbox in field_bboxes.items()
    }


def compute_calibrated_confidence(
    model_signal: float,
    clarity_factor: float,
    format_valid: bool = True,
) -> float:
    """Calculate composite field confidence grounded in physical image quality and format checks.

    Formula: Field Confidence = Model Extraction Signal * Local Clarity Factor * Format Validity Factor
    """
    format_factor = 1.0 if format_valid else 0.0
    composite = model_signal * clarity_factor * format_factor
    return float(np.clip(composite, 0.0, 1.0))



