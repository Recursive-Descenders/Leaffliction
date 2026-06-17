"""xfm CLI: parse flags and run image transformations."""

from dataclasses import dataclass
from pathlib import Path

import typer

from transformation.util import (
    validate_dst_is_directory,
    validate_source_exists,
)

DEFAULT_SOURCE = Path("leaves/images")
DEFAULT_DST = Path("outputs/transformation")

TRANSFORM_FLAGS = (
    "blur",
    "mask",
    "lesion_analysis",
    "roi",
    "analyze",
    "pseudolandmarks",
    "histogram",
)


@dataclass(frozen=True)
class ParsedArgs:
    src: Path
    file: str | None
    dst: Path | None
    transforms: frozenset[str]


def _resolve_source(path: Path) -> tuple[Path, str | None]:
    validate_source_exists(path)
    if path.is_file():
        return path.parent, path.name
    return path, None


def _resolve_destination(path: Path) -> Path:
    validate_dst_is_directory(path)
    return path


def _to_parsed_args(
    *,
    src: Path,
    dst: Path | None,
    all_transforms: bool,
    blur: bool,
    mask: bool,
    lesion_shape: bool,
    roi: bool,
    analyze: bool,
    pseudolandmarks: bool,
    histogram: bool,
) -> ParsedArgs:
    resolved_src, file = _resolve_source(src)
    resolved_dst = _resolve_destination(dst) if dst is not None else None

    selected = {
        name
        for name, enabled in (
            ("blur", blur),
            ("mask", mask),
            ("lesion_analysis", lesion_shape),
            ("roi", roi),
            ("analyze", analyze),
            ("pseudolandmarks", pseudolandmarks),
            ("histogram", histogram),
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
        help=(
            "Source image file or directory "
            f"(default: {DEFAULT_SOURCE}). "
            "Examples: "
            "'uv run xfm -s leaf.jpg' (preview), "
            "'uv run xfm -s leaves/images -d out/ -m -g' (batch)"
        ),
    ),
    dst: Path | None = typer.Option(
        None,
        "-d",
        "--dst",
        help=(
            "Destination directory for batch outputs "
            f"(default: {DEFAULT_DST}/<transform>). "
            "Omit with a single source image to show previews."
        ),
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
        help="Gaussian blur",
    ),
    mask: bool = typer.Option(
        False,
        "-m",
        "--mask",
        help="Leaf mask contour overlay (cached)",
    ),
    lesion_shape: bool = typer.Option(
        False,
        "-ls",
        "--lesion-shape",
        help=(
            "Lesion analysis: spot-mask contours + summary "
            "(uses cached leaf and spot masks)"
        ),
    ),
    roi: bool = typer.Option(
        False,
        "-r",
        "--roi",
        help="ROI visualization",
    ),
    analyze: bool = typer.Option(
        False,
        "-anlz",
        "--analyze",
        help="PlantCV size analysis overlay",
    ),
    pseudolandmarks: bool = typer.Option(
        False,
        "-pl",
        "--pseudolandmarks",
        help="Pseudolandmarks overlay",
    ),
    histogram: bool = typer.Option(
        False,
        "-g",
        "--histogram",
        help=(
            "Per-image color histogram from cached leaf and spot masks"
        ),
    ),
) -> None:
    try:
        args = _to_parsed_args(
            src=src,
            dst=dst,
            all_transforms=all_transforms,
            blur=blur,
            mask=mask,
            lesion_shape=lesion_shape,
            roi=roi,
            analyze=analyze,
            pseudolandmarks=pseudolandmarks,
            histogram=histogram,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    from Transformation import run_transformations

    run_transformations(
        src=args.src,
        dst=args.dst,
        file=args.file,
        transforms=args.transforms,
    )
