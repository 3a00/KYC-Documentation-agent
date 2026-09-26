"""CLI utility to generate augmented synthetic document packages for demo testing.

Exports ready-to-test sample sets (clean, specular glare, gaussian blur, identity mismatch)
to a local directory (default: demo_samples/) for immediate drag-and-drop verification.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

# Ensure project root is on sys.path when invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.benchmark_suite import BenchmarkScenario, generate_synthetic_benchmark_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def export_demo_samples(output_dir: Path, seed: int = 42) -> None:
    """Generate and export document packages across all key demonstration scenarios."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating synthetic packages for demo testing (seed=%d)...", seed)

    # Generate 1 package per scenario
    packages = generate_synthetic_benchmark_split(sample_count_per_scenario=1, seed=seed)

    scenario_folder_names = {
        BenchmarkScenario.CLEAN: "01_clean_autopass",
        BenchmarkScenario.SPECULAR_GLARE: "02_specular_glare_escalate",
        BenchmarkScenario.BLURRED: "03_gaussian_blur_escalate",
        BenchmarkScenario.TILTED: "04_tilted_autopass",
        BenchmarkScenario.IDENTITY_MISMATCH: "05_identity_mismatch_reject",
    }

    for pkg in packages:
        folder_name = scenario_folder_names.get(pkg.scenario, pkg.scenario.value)
        scenario_dir = output_dir / folder_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Save document images
        nid_path = scenario_dir / "national_id.png"
        biz_path = scenario_dir / "business_license.png"
        tax_path = scenario_dir / "tax_card.png"
        gt_path = scenario_dir / "ground_truth.json"

        pkg.national_id_image.save(nid_path, format="PNG")
        pkg.business_license_image.save(biz_path, format="PNG")
        pkg.tax_card_image.save(tax_path, format="PNG")

        metadata: dict[str, Any] = {
            "package_id": pkg.package_id,
            "scenario": pkg.scenario.value,
            "expected_outcome": pkg.expected_outcome.value,
            "ground_truth": pkg.ground_truth,
            "planted_defects": pkg.planted_defects,
        }

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info("Exported [%s] -> %s", pkg.scenario.value, scenario_dir)

    logger.info("Successfully exported %d sample suites to %s", len(packages), output_dir.resolve())


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Export augmented synthetic document packages for demo testing.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_samples"),
        help="Destination directory for exported document samples (default: demo_samples).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible document synthesis (default: 42).",
    )
    args = parser.parse_args()
    export_demo_samples(output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
