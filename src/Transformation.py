from collections.abc import Callable
from pathlib import Path

import typer

from transformation.analyze import analyze
from transformation.color_histogram import histogram
from transformation.cli import DEFAULT_DST, run
from transformation.lesion_analysis import lesion_analysis
from transformation.gaussian_blur import gaussian_blur
from transformation.mask import mask
from transformation.pseudolandmarks import pseudolandmarks
from transformation.preview import show_transformations
from transformation.roi import roi

BLUR_OUTPUT_DIR = DEFAULT_DST / "gaussian_blur"
MASK_OUTPUT_DIR = DEFAULT_DST / "mask"
ROI_OUTPUT_DIR = DEFAULT_DST / "roi"
ANALYZE_OUTPUT_DIR = DEFAULT_DST / "analyze"
PSEUDOLANDMARKS_OUTPUT_DIR = DEFAULT_DST / "pseudolandmarks"
HISTOGRAM_OUTPUT_DIR = DEFAULT_DST / "histogram"
LESION_ANALYSIS_OUTPUT_DIR = DEFAULT_DST / "lesion_analysis"

TransformFn = Callable[..., list[Path]]

TRANSFORMS: dict[str, tuple[TransformFn, Path]] = {
    "blur": (gaussian_blur, BLUR_OUTPUT_DIR),
    "mask": (mask, MASK_OUTPUT_DIR),
    "roi": (roi, ROI_OUTPUT_DIR),
    "analyze": (analyze, ANALYZE_OUTPUT_DIR),
    "pseudolandmarks": (pseudolandmarks, PSEUDOLANDMARKS_OUTPUT_DIR),
    "histogram": (histogram, HISTOGRAM_OUTPUT_DIR),
    "lesion_analysis": (lesion_analysis, LESION_ANALYSIS_OUTPUT_DIR),
}
TRANSFORM_ORDER = (
    "blur",
    "mask",
    "lesion_analysis",
    "roi",
    "analyze",
    "pseudolandmarks",
    "histogram",
)


def run_transformations(
    *,
    src: Path,
    dst: Path | None = None,
    file: str | Path | None = None,
    transforms: frozenset[str] | None = None,
) -> None:
    selected = transforms or frozenset(TRANSFORMS)

    if file is not None and dst is None:
        show_transformations(
            src=src,
            file=str(file),
            transforms=selected,
            order=TRANSFORM_ORDER,
        )
        return

    for name in TRANSFORM_ORDER:
        if name not in selected:
            continue

        transform_fn, default_dst = TRANSFORMS[name]
        output_dir = dst if dst is not None else default_dst
        transform_fn(
            src=src,
            dst=output_dir,
            file=file,
        )


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
