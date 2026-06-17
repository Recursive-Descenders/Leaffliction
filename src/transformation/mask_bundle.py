"""Single-pass batch processing for mask-related transforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import matplotlib
import matplotlib.pyplot as plt

from transformation.color_histogram import build_color_histogram_figure
from transformation.leaf_cache import get_leaf_mask
from transformation.lesion_analysis import build_lesion_analysis_figure
from transformation.lesion_cache import ensure_lesion_record
from transformation.mask import _overlay_from_record
from transformation.parallel import parallel_map
from transformation.util import (
    get_image_paths,
    make_output_path,
    skip_image,
    validate_image_readable,
)

MASK_BUNDLE_TRANSFORMS = frozenset({
    "mask",
    "lesion_analysis",
    "histogram",
})


@dataclass(frozen=True)
class MaskBundleWorkItem:
    src: str
    dst_dirs: tuple[tuple[str, str], ...]
    file: str | None
    image_path: str
    selected: tuple[str, ...]


def process_mask_bundle_image(
    item: MaskBundleWorkItem,
) -> dict[str, str]:
    matplotlib.use("Agg")

    src = Path(item.src)
    image_path = Path(item.image_path)
    file = item.file
    selected = frozenset(item.selected)
    dst_by_name = dict(item.dst_dirs)

    try:
        image = validate_image_readable(image_path)
    except ValueError as exc:
        skip_image(image_path, str(exc))
        return {}

    record = get_leaf_mask(image_path, image=image)
    if record is None:
        skip_image(image_path, "no detected leaf")
        return {}

    lesion_record = None
    if selected & {"lesion_analysis", "histogram"}:
        lesion_record = ensure_lesion_record(image_path, image=image)
        if lesion_record is None:
            skip_image(image_path, "no spot mask")
            return {}

    outputs: dict[str, str] = {}

    if "mask" in selected:
        overlay = _overlay_from_record(image, record)
        if overlay is None:
            skip_image(image_path, "no detected leaf")
            return {}
        output_file = make_output_path(
            src,
            Path(dst_by_name["mask"]),
            image_path,
            file,
            "mask",
        )
        cv2.imwrite(str(output_file), overlay)
        outputs["mask"] = str(output_file)

    if "lesion_analysis" in selected and lesion_record is not None:
        figure = build_lesion_analysis_figure(
            image,
            record.mask,
            lesion_record,
        )
        output_file = make_output_path(
            src,
            Path(dst_by_name["lesion_analysis"]),
            image_path,
            file,
            "lesion_analysis",
            extension=".png",
        )
        figure.savefig(output_file, dpi=160, bbox_inches="tight")
        plt.close(figure)
        outputs["lesion_analysis"] = str(output_file)

    if "histogram" in selected and lesion_record is not None:
        figure = build_color_histogram_figure(
            image,
            record.mask,
            lesion_record.spot_mask,
        )
        output_file = make_output_path(
            src,
            Path(dst_by_name["histogram"]),
            image_path,
            file,
            "histogram",
            extension=".png",
        )
        figure.savefig(output_file, dpi=160, bbox_inches="tight")
        plt.close(figure)
        outputs["histogram"] = str(output_file)

    return outputs


def mask_bundle(
    src: str | Path,
    output_dirs: dict[str, Path],
    file: str | Path | None,
    selected: frozenset[str],
    *,
    jobs: int = 0,
) -> list[Path]:
    src = Path(src)
    file_name = str(file) if file is not None else None
    selected_names = tuple(sorted(selected))
    dst_dirs = tuple(
        (name, str(output_dirs[name]))
        for name in selected_names
    )

    items = [
        MaskBundleWorkItem(
            src=str(src),
            dst_dirs=dst_dirs,
            file=file_name,
            image_path=str(image_path),
            selected=selected_names,
        )
        for image_path in get_image_paths(src, file)
    ]
    results = parallel_map(
        process_mask_bundle_image,
        items,
        jobs=jobs,
        desc="mask_bundle",
    )

    saved_paths: list[Path] = []
    for result in results:
        if not result:
            continue
        saved_paths.extend(Path(path) for path in result.values())
    return saved_paths
