import argparse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SOURCE = Path("leaves/images")
DEFAULT_DST = Path("outputs/transformation")

TRANSFORM_FLAGS = (
    "blur",
    "mask",
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
    if path.is_file():
        return path.parent, path.name

    if path.is_dir():
        return path, None

    raise FileNotFoundError(f"Source path does not exist: {path}")


def _resolve_destination(path: Path) -> Path:
    if path.is_file():
        return path.parent

    return path


def parse_args(argv: list[str] | None = None) -> ParsedArgs:
    parser = argparse.ArgumentParser(
        description="Apply image transformations to leaf photos.",
    )
    parser.add_argument(
        "-src",
        dest="src",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Source image file or directory "
        f"(default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "-dst",
        dest="dst",
        type=Path,
        default=None,
        help="Destination directory for outputs "
        f"(default: {DEFAULT_DST}/<transform>)",
    )
    parser.add_argument(
        "-all",
        dest="all_transforms",
        action="store_true",
        help="Run all transformations (default when none are selected)",
    )

    for name in TRANSFORM_FLAGS:
        parser.add_argument(
            f"-{name}",
            dest=name,
            action="store_true",
            help=f"Run the {name} transformation",
        )

    args = parser.parse_args(argv)

    src, file = _resolve_source(args.src)
    dst = (
        _resolve_destination(args.dst)
        if args.dst is not None
        else None
    )

    selected = {
        name
        for name in TRANSFORM_FLAGS
        if getattr(args, name)
    }

    if not selected or args.all_transforms:
        selected = set(TRANSFORM_FLAGS)

    return ParsedArgs(
        src=src,
        file=file,
        dst=dst,
        transforms=frozenset(selected),
    )
