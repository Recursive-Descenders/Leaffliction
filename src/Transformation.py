import importlib.util
from collections.abc import Callable
from pathlib import Path
from transformation.analyze import analyze
from transformation.gaussian_blur import gaussian_blur
from transformation.histogram import histogram
from transformation.mask import mask
from transformation.pseudolandmarks import pseudolandmarks
from transformation.roi import roi

_ARG_PARSER_PATH = (
    Path(__file__).parent / "transformation" / "arg-parser.py"
)
_spec = importlib.util.spec_from_file_location(
    "transformation.arg_parser",
    _ARG_PARSER_PATH,
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load arg parser from {_ARG_PARSER_PATH}")

_arg_parser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_arg_parser)
parse_args = _arg_parser.parse_args

DEFAULT_DST = _arg_parser.DEFAULT_DST

BLUR_OUTPUT_DIR = DEFAULT_DST / "gaussian_blur"
MASK_OUTPUT_DIR = DEFAULT_DST / "mask"
ROI_OUTPUT_DIR = DEFAULT_DST / "roi"
ANALYZE_OUTPUT_DIR = DEFAULT_DST / "analyze"
PSEUDOLANDMARKS_OUTPUT_DIR = DEFAULT_DST / "pseudolandmarks"
HISTOGRAM_OUTPUT_DIR = DEFAULT_DST / "histogram"

TransformFn = Callable[..., list[Path]]

TRANSFORMS: dict[str, tuple[TransformFn, Path]] = {
    "blur": (gaussian_blur, BLUR_OUTPUT_DIR),
    "mask": (mask, MASK_OUTPUT_DIR),
    "roi": (roi, ROI_OUTPUT_DIR),
    "analyze": (analyze, ANALYZE_OUTPUT_DIR),
    "pseudolandmarks": (pseudolandmarks, PSEUDOLANDMARKS_OUTPUT_DIR),
    "histogram": (histogram, HISTOGRAM_OUTPUT_DIR),
}


def run_transformations(
    *,
    src: Path,
    dst: Path | None = None,
    file: str | Path | None = None,
    transforms: frozenset[str] | None = None,
) -> None:
    selected = transforms or frozenset(TRANSFORMS)

    for name in TRANSFORMS:
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
    args = parse_args()
    run_transformations(
        src=args.src,
        dst=args.dst,
        file=args.file,
        transforms=args.transforms,
    )


if __name__ == "__main__":
    main()
