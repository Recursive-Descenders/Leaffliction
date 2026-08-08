from collections.abc import Callable
from pathlib import Path

import typer

from transformation.analyze import analyze
from transformation.color_histogram import histogram
from transformation.cli import DEFAULT_DST, run
from transformation.lesion_analysis import lesion_analysis
from transformation.gaussian_blur import gaussian_blur
from transformation.mask import mask
from transformation.mask_bundle import MASK_BUNDLE_TRANSFORMS, mask_bundle
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
PARALLEL_TRANSFORMS = MASK_BUNDLE_TRANSFORMS


def _output_dir(
    name: str,
    *,
    dst: Path | None,
    default_dst: Path,
) -> Path:
    return dst if dst is not None else default_dst


def run_transformations(
    *,
    src: Path,
    dst: Path | None = None,
    file: str | Path | None = None,
    transforms: frozenset[str] | None = None,
    jobs: int = 0,
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

    bundle = selected & MASK_BUNDLE_TRANSFORMS
    handled: set[str] = set()

    if len(bundle) >= 2:
        output_dirs = {
            name: _output_dir(name, dst=dst, default_dst=TRANSFORMS[name][1])
            for name in bundle
        }
        mask_bundle(
            src=src,
            output_dirs=output_dirs,
            file=file,
            selected=bundle,
            jobs=jobs,
        )
        handled = set(bundle)
    elif len(bundle) == 1:
        name = next(iter(bundle))
        transform_fn, default_dst = TRANSFORMS[name]
        transform_fn(
            src=src,
            dst=_output_dir(name, dst=dst, default_dst=default_dst),
            file=file,
            jobs=jobs,
        )
        handled = {name}

    for name in TRANSFORM_ORDER:
        if name not in selected or name in handled:
            continue

        transform_fn, default_dst = TRANSFORMS[name]
        output_dir = _output_dir(name, dst=dst, default_dst=default_dst)
        if name in PARALLEL_TRANSFORMS:
            transform_fn(
                src=src,
                dst=output_dir,
                file=file,
                jobs=jobs,
            )
            continue

        transform_fn(
            src=src,
            dst=output_dir,
            file=file,
        )


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
