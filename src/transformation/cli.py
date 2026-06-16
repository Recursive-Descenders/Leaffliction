from dataclasses import dataclass
from pathlib import Path

import typer

DEFAULT_SOURCE = Path("leaves/images")
DEFAULT_DST = Path("outputs/transformation")

TRANSFORM_FLAGS = (
    "blur",
    "mask",
    "roi",
    "analyze",
    "pseudolandmarks",
    "histogram",
    "spot_mask",
)


@dataclass(frozen=True)
class ParsedArgs:
    src: Path
    file: str | None
    dst: Path | None
    transforms: frozenset[str]


def _resolve_source(path: Path) -> tuple[Path, str | None]:
    if path.is_file():
        return path.parent, path.name

    if path.is_dir():
        return path, None

    raise FileNotFoundError(f"Source path does not exist: {path}")


def _resolve_destination(path: Path) -> Path:
    if path.is_file():
        return path.parent

    return path


def _to_parsed_args(
    *,
    src: Path,
    dst: Path | None,
    all_transforms: bool,
    blur: bool,
    mask: bool,
    roi: bool,
    analyze: bool,
    pseudolandmarks: bool,
    histogram: bool,
    spot_mask: bool,
) -> ParsedArgs:
    resolved_src, file = _resolve_source(src)
    resolved_dst = _resolve_destination(dst) if dst is not None else None

    selected = {
        name
        for name, enabled in (
            ("blur", blur),
            ("mask", mask),
            ("roi", roi),
            ("analyze", analyze),
            ("pseudolandmarks", pseudolandmarks),
            ("histogram", histogram),
            ("spot_mask", spot_mask),
        )
        if enabled
    }

    if not selected or all_transforms:
        selected = set(TRANSFORM_FLAGS)

    return ParsedArgs(
        src=resolved_src,
        file=file,
        dst=resolved_dst,
        transforms=frozenset(selected),
    )


def run(
    src: Path = typer.Option(
        DEFAULT_SOURCE,
        "-s",
        "--src",
        help="Source image file or directory "
        f"(default: {DEFAULT_SOURCE})",
    ),
    dst: Path | None = typer.Option(
        None,
        "-d",
        "--dst",
        help="Destination directory for outputs "
        f"(default: {DEFAULT_DST}/<transform>; "
        "omit with a single source image to show previews)",
    ),
    all_transforms: bool = typer.Option(
        False,
        "-a",
        "--all",
        help="Run all transformations (default when none are selected)",
    ),
    blur: bool = typer.Option(
        False,
        "-b",
        "--blur",
        help="Run the blur transformation",
    ),
    mask: bool = typer.Option(
        False,
        "-m",
        "--mask",
        help="Run the mask transformation",
    ),
    roi: bool = typer.Option(
        False,
        "-r",
        "--roi",
        help="Run the roi transformation",
    ),
    analyze: bool = typer.Option(
        False,
        "-anlz",
        "--analyze",
        help="Run the analyze transformation",
    ),
    pseudolandmarks: bool = typer.Option(
        False,
        "-pl",
        "--pseudolandmarks",
        help="Run the pseudolandmarks transformation",
    ),
    histogram: bool = typer.Option(
        False,
        "-g",
        "--histogram",
        help="Run the histogram transformation",
    ),
    spot_mask: bool = typer.Option(
        False,
        "-sm",
        "--spot-mask",
        help="Run the spot mask transformation",
    ),
) -> None:
    args = _to_parsed_args(
        src=src,
        dst=dst,
        all_transforms=all_transforms,
        blur=blur,
        mask=mask,
        roi=roi,
        analyze=analyze,
        pseudolandmarks=pseudolandmarks,
        histogram=histogram,
        spot_mask=spot_mask,
    )

    from Transformation import run_transformations

    run_transformations(
        src=args.src,
        dst=args.dst,
        file=args.file,
        transforms=args.transforms,
    )
