"""
Interactive matplotlib viewer for geometric augmentations.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.widgets import RadioButtons, Slider

from augmentation.geometric import (
    apply_crop,
    apply_radial_distortion,
    apply_flip,
    apply_rotate,
    apply_shear,
    apply_skew,
)
from augmentation.util import load_image, make_test_image


@dataclass(frozen=True)
class ParamSpec:
    """One slider: its keyword name plus the range the slider sweeps."""

    name: str
    label: str
    min: float
    max: float
    default: float
    step: float


@dataclass(frozen=True)
class Method:
    """An augmentation: a callable plus the sliders feeding its parameters."""

    key: str
    label: str
    func: Callable[..., np.ndarray]
    params: list[ParamSpec] = field(default_factory=list)


METHODS: list[Method] = [
    Method("flip", "Flip", apply_flip),
    Method(
        "rotate",
        "Rotate",
        apply_rotate,
        [ParamSpec("angle", "Angle (deg)", -180, 180, 25, 1)],
    ),
    Method(
        "skew",
        "Skew",
        apply_skew,
        [
            ParamSpec("skx", "skx", -0.4, 0.4, 0.15, 0.01),
            ParamSpec("sky", "sky", -0.4, 0.4, 0.15, 0.01),
        ],
    ),
    Method(
        "shear",
        "Shear",
        apply_shear,
        [
            ParamSpec("kx", "kx", -0.5, 0.5, 0.2, 0.01),
            ParamSpec("ky", "ky", -0.5, 0.5, 0.0, 0.01),
        ],
    ),
    Method(
        "crop",
        "Crop",
        apply_crop,
        [
            ParamSpec("scale", "scale", 0.1, 1.0, 0.5, 0.01),
        ],
    ),
    Method(
        "distortion",
        "Distortion",
        apply_radial_distortion,
        [
            ParamSpec("k", "k", 0.0, 0.5, 0.3, 0.01),
        ],
    ),
]


class AugmentationViewer:
    """Wire the radio buttons, dynamic sliders, and the two image panels."""

    SLIDER_TOP = 0.30
    SLIDER_LEFT = 0.45
    SLIDER_WIDTH = 0.45
    SLIDER_HEIGHT = 0.03
    SLIDER_GAP = 0.07

    def __init__(self, image: np.ndarray, image_name: str) -> None:
        self.image = image
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.figure = plt.figure(figsize=(11, 7))
        self.figure.suptitle(f"Augmentation preview — {image_name}")

        self.ax_original = self.figure.add_axes((0.05, 0.45, 0.4, 0.5))
        self.ax_augmented = self.figure.add_axes((0.55, 0.45, 0.4, 0.5))
        for axis in (self.ax_original, self.ax_augmented):
            axis.axis("off")
        self.ax_original.set_title("Original")
        self.ax_original.imshow(self.image_rgb)
        self.augmented_artist = self.ax_augmented.imshow(self.image_rgb)

        self.ax_radio = self.figure.add_axes((0.05, 0.05, 0.22, 0.32))
        self.ax_radio.set_title("Method", fontsize=10)
        self.radio = RadioButtons(
            self.ax_radio, [method.label for method in METHODS]
        )
        self.radio.on_clicked(self._on_method_label)  # type: ignore

        # Placeholder text shown when the selected method is not implemented.
        self.message = self.figure.text(
            self.SLIDER_LEFT, self.SLIDER_TOP, "", color="#b00020", fontsize=12
        )

        self.sliders: list[Slider] = []
        self.current_method = METHODS[0]
        self._select_method(self.current_method)

    def _on_method_label(self, label: str) -> None:
        method = next(m for m in METHODS if m.label == label)
        self._select_method(method)

    def _clear_sliders(self) -> None:
        for slider in self.sliders:
            slider.ax.remove()
        self.sliders = []

    def _select_method(self, method: Method) -> None:
        self.current_method = method
        self._clear_sliders()
        self.message.set_text("")

        defaults = {p.name: p.default for p in method.params}
        result = self._safe_apply(method, defaults)
        if result is None:
            # try-call hit NotImplementedError: show placeholder, no sliders.
            self.message.set_text(f"{method.label}: not implemented yet")
            self.augmented_artist.set_data(self.image_rgb)
            self.ax_augmented.set_title("Augmented (—)")
            self.figure.canvas.draw_idle()
            return

        for index, spec in enumerate(method.params):
            bottom = self.SLIDER_TOP - index * self.SLIDER_GAP
            rect = (
                self.SLIDER_LEFT,
                bottom,
                self.SLIDER_WIDTH,
                self.SLIDER_HEIGHT,
            )
            slider_ax = self.figure.add_axes(rect)
            slider = Slider(
                slider_ax,
                spec.label,
                spec.min,
                spec.max,
                valinit=spec.default,
                valstep=spec.step,
            )
            slider.on_changed(lambda _value: self._render())
            self.sliders.append(slider)

        self._render(initial=result)

    def _current_values(self) -> dict[str, float]:
        return {
            spec.name: slider.val
            for spec, slider in zip(self.current_method.params, self.sliders)
        }

    def _safe_apply(
        self, method: Method, values: dict[str, float]
    ) -> np.ndarray | None:
        """Apply ``method``; return ``None`` if it is not implemented yet."""
        try:
            return method.func(self.image, **values)
        except NotImplementedError:
            return None

    def _render(self, initial: np.ndarray | None = None) -> None:
        result = initial
        if result is None:
            result = self._safe_apply(
                self.current_method, self._current_values()
            )
        if result is None:
            return
        self.augmented_artist.set_data(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        self.ax_augmented.set_title(f"Augmented ({self.current_method.label})")
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def run(
    image_path: Path = typer.Argument(
        None,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Image to preview augmentations on.",
    ),
) -> None:
    if image_path is None:
        image = make_test_image()
        name = "synthetic test pattern"
    else:
        image = load_image(image_path)
        name = image_path.name
    AugmentationViewer(image, name).show()


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
