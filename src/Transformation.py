from pathlib import Path
from transformation.analyze import analyze
from transformation.gaussian_blur import gaussian_blur
from transformation.histogram import histogram
from transformation.mask import mask
from transformation.pseudolandmarks import pseudolandmarks
from transformation.roi import roi


SOURCE_DIR = Path("leaves/images")
BLUR_OUTPUT_DIR = Path("outputs/transformation/gaussian_blur")
MASK_OUTPUT_DIR = Path("outputs/transformation/mask")
ROI_OUTPUT_DIR = Path("outputs/transformation/roi")
ANALYZE_OUTPUT_DIR = Path("outputs/transformation/analyze")
PSEUDOLANDMARKS_OUTPUT_DIR = Path("outputs/transformation/pseudolandmarks")
HISTOGRAM_OUTPUT_DIR = Path("outputs/transformation/histogram")


def main() -> None:
    gaussian_blur(
        src=SOURCE_DIR,
        dst=BLUR_OUTPUT_DIR,
    )
    mask(
        src=SOURCE_DIR,
        dst=MASK_OUTPUT_DIR,
    )
    roi(
        src=SOURCE_DIR,
        dst=ROI_OUTPUT_DIR,
    )
    analyze(
        src=SOURCE_DIR,
        dst=ANALYZE_OUTPUT_DIR,
    )
    pseudolandmarks(
        src=SOURCE_DIR,
        dst=PSEUDOLANDMARKS_OUTPUT_DIR,
    )
    histogram(
        src=SOURCE_DIR,
        dst=HISTOGRAM_OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
