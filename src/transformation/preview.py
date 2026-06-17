import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import matplotlib
import numpy as np  # type: ignore[import-not-found]
from transformation.analyze import apply_analyze
from transformation.color_histogram import build_color_histogram_figure
from transformation.lesion_analysis import build_lesion_analysis_figure
from transformation.lesion_cache import ensure_lesion_record
from transformation.leaf_cache import get_leaf_mask
from transformation.gaussian_blur import apply_gaussian_blur
from transformation.mask import apply_mask
from transformation.pseudolandmarks import apply_pseudolandmarks
from transformation.roi import apply_roi
from transformation.util import get_image_paths, validate_image_readable

_INTERACTIVE_BACKENDS = (
    "TkAgg",
    "Qt5Agg",
    "GTK4Agg",
    "GTK3Agg",
    "wxAgg",
)


def _configure_matplotlib() -> None:
    if os.environ.get("MPLBACKEND"):
        return

    for backend in _INTERACTIVE_BACKENDS:
        try:
            matplotlib.use(backend, force=False)
            return
        except ImportError:
            continue


_configure_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402

PreviewPanel = np.ndarray

PANEL_TITLES: dict[str, str] = {
    "blur": "Gaussian blur",
    "mask": "Mask",
    "roi": "ROI",
    "analyze": "Analyze",
    "pseudolandmarks": "Pseudolandmarks",
    "histogram": "Color histogram",
    "lesion_analysis": "Lesion analysis",
}

PANEL_COMPUTERS: dict[str, Callable[[np.ndarray], PreviewPanel | None]] = {
    "blur": apply_gaussian_blur,
}


@dataclass(frozen=True)
class PreviewResult:
    title: str
    panel: PreviewPanel


PATH_AWARE_PANELS = frozenset({
    "mask",
    "roi",
    "analyze",
    "pseudolandmarks",
    "histogram",
    "lesion_analysis",
})


def _figure_to_bgr(figure: plt.Figure) -> np.ndarray:
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    buffer = np.frombuffer(
        figure.canvas.buffer_rgba(),
        dtype=np.uint8,
    ).reshape(height, width, 4)
    return cv2.cvtColor(buffer[:, :, :3], cv2.COLOR_RGB2BGR)


def _figure_panel(figure: plt.Figure) -> np.ndarray:
    panel = _figure_to_bgr(figure)
    plt.close(figure)
    return panel


def _compute_panel(
    name: str,
    image: np.ndarray,
    image_path: Path,
) -> PreviewPanel | None:
    if name == "mask":
        return apply_mask(image, image_path=image_path)
    if name == "roi":
        return apply_roi(image, image_path=image_path)
    if name == "analyze":
        return apply_analyze(image, image_path=image_path)
    if name == "pseudolandmarks":
        return apply_pseudolandmarks(image, image_path=image_path)
    if name == "histogram":
        record = get_leaf_mask(image_path)
        if record is None:
            return None
        lesion_record = ensure_lesion_record(image_path)
        if lesion_record is None:
            return None
        return _figure_panel(
            build_color_histogram_figure(
                image,
                record.mask,
                lesion_record.spot_mask,
            )
        )
    if name == "lesion_analysis":
        record = get_leaf_mask(image_path)
        if record is None:
            return None
        lesion_record = ensure_lesion_record(image_path)
        if lesion_record is None:
            return None
        return _figure_panel(
            build_lesion_analysis_figure(
                image,
                record.mask,
                lesion_record,
            )
        )

    return PANEL_COMPUTERS[name](image)


def _collect_panels(
    image: np.ndarray,
    image_path: Path,
    transforms: frozenset[str],
    order: tuple[str, ...],
) -> list[PreviewResult]:
    panels: list[PreviewResult] = [
        PreviewResult(title="Original", panel=image.copy()),
    ]

    for name in order:
        if name not in transforms:
            continue

        if name in PATH_AWARE_PANELS:
            panel = _compute_panel(name, image, image_path)
        else:
            panel = PANEL_COMPUTERS[name](image)
        if panel is None:
            print(f"Skipped {name}: no result for preview")
            continue

        panels.append(PreviewResult(title=PANEL_TITLES[name], panel=panel))

    return panels


def _screen_size() -> tuple[int, int]:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size
    except Exception:
        pass

    if sys.platform.startswith("linux"):
        try:
            output = subprocess.check_output(
                ["xdpyinfo"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in output.splitlines():
                if "dimensions:" not in line:
                    continue
                dimensions = line.split("dimensions:")[1].strip().split()[0]
                width, height = dimensions.split("x")
                return int(width), int(height)
        except (
            subprocess.SubprocessError,
            OSError,
            ValueError,
        ):
            pass

    return 1920, 1080


def _preview_layout(
    panel_count: int,
    margin: float = 0.96,
    dpi: int = 100,
) -> tuple[int, int, tuple[float, float], int]:
    cols = min(3, panel_count)
    rows = (panel_count + cols - 1) // cols
    screen_w, screen_h = _screen_size()
    figsize = (
        (screen_w * margin) / dpi,
        (screen_h * margin) / dpi,
    )
    return cols, rows, figsize, dpi


def _fit_to_screen(
    image: np.ndarray,
    margin: float = 0.96,
) -> np.ndarray:
    screen_w, screen_h = _screen_size()
    max_w = int(screen_w * margin)
    max_h = int(screen_h * margin)
    height, width = image.shape[:2]

    scale = min(max_w / width, max_h / height, 1.0)
    if scale >= 1.0:
        return image

    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def _show_with_cv2(figure: plt.Figure, title: str) -> None:
    window = f"Leaffliction - {title}"
    preview = _fit_to_screen(_figure_to_bgr(figure))
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, preview)
    cv2.resizeWindow(window, preview.shape[1], preview.shape[0])
    print("Press any key in the preview window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _open_with_system_viewer(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
        return

    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
        return

    subprocess.run(["xdg-open", str(path)], check=False)


def _display_figure(figure: plt.Figure, title: str) -> None:
    backend = matplotlib.get_backend().lower()
    if "agg" not in backend:
        plt.show()
        return

    if os.environ.get("DISPLAY") or sys.platform.startswith("win"):
        _show_with_cv2(figure, title)
        plt.close(figure)
        return

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    ) as tmp:
        output = Path(tmp.name)

    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"No display available; preview saved to {output}")
    _open_with_system_viewer(output)


def show_transformations(
    *,
    src: Path,
    file: str,
    transforms: frozenset[str],
    order: tuple[str, ...],
) -> None:
    image_path = get_image_paths(src, file)[0]
    image = validate_image_readable(image_path)

    panels = _collect_panels(image, image_path, transforms, order)
    if not panels:
        print("No transformations to display.")
        return

    count = len(panels)
    cols, rows, figsize, dpi = _preview_layout(count)
    figure, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
    )
    figure.suptitle(image_path.name)

    flat_axes = axes.flatten()

    for axis in flat_axes[len(panels):]:
        axis.axis("off")

    for axis, result in zip(flat_axes, panels, strict=False):
        axis.set_title(result.title)
        axis.axis("off")

        axis.imshow(cv2.cvtColor(result.panel, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    _display_figure(figure, image_path.name)
