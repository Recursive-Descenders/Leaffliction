from pathlib import Path
from transformation.gaussian_blur import gaussian_blur


SOURCE_DIR = Path("leaves/images")
OUTPUT_DIR = Path("outputs/transformation/gaussian_blur")


def main() -> None:
    gaussian_blur(
        src=SOURCE_DIR,
        dst=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
