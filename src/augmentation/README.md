# Augmentation

Image augmentation module for the **Leaffliction** 42 project.

## Coverage (what currently works)

The augmentation pipeline lives in `src/augmentation/` and is driven by a
registry (`METHODS` in `visualization.py`). The CLI entry is
`src/Augmentation.py`.

Implemented geometric transforms in `src/augmentation/geometric.py`:

- `apply_flip` — horizontal mirror
- `apply_rotate(angle=25.0)` — rotate, empty corners filled white
- `apply_skew(skr=0.2, skc=0.2)` — affine skew, white border fill

IO helpers in `src/augmentation/util.py`:

- `load_image`
- `build_output_path` — naming rule `<stem>_<Method><suffix>`
- `save_image`

Interactive matplotlib preview viewer in `src/augmentation/visualization.py`
(`AugmentationViewer`) with radio buttons and sliders.

## TODO (not implemented yet)

These three raise `NotImplementedError` (Stage 3) and are skipped by the CLI
until implemented:

- `apply_shear(factor=0.2)`
- `apply_crop(scale=0.8)`
- `apply_distortion(strength=8.0)`

> Note: the 42 subject's minimum requirement is 6 augmentation types; only 3
> are implemented today, so implementing these three is what reaches the
> subject minimum.

## How to run

The CLI applies every *implemented* augmentation to one image and writes each
result to an output directory (default `data/augmented_directory`) using the
naming rule `<stem>_<Method><suffix>` (e.g. `image (1)_Flip.JPG`).
Unimplemented methods are skipped automatically. It saves only — it does not
display.

```bash
uv run aug "data/raw/Apple/apple_healthy/image (1).JPG"
# or directly (src/ is on sys.path when the script runs):
./src/Augmentation.py "data/raw/Apple/apple_healthy/image (1).JPG"
# custom output dir:
uv run aug --dst data/augmented_directory "data/raw/Apple/apple_healthy/image (1).JPG"
```

The `aug` console script is defined in `pyproject.toml` under
`[project.scripts]` (`aug = "Augmentation:main"`).

## Interactive visualization

`augmentation/visualization.py` opens a matplotlib window with the original on
the left and the augmented result on the right. The radio buttons switch
between methods; the sliders (rebuilt per method from `METHODS`) tweak that
method's parameters live. Methods that raise `NotImplementedError` show a red
"not implemented yet" placeholder instead of sliders.

```bash
uv run python -m augmentation.visualization "data/raw/Apple_healthy/image (1).JPG"
```

Notes:

- Requires a GUI backend — run from a desktop session, not a headless shell.
- Preview only; it does not save. Use `uv run aug ...` to write files.
- To add a new method, append a `Method(...)` entry to `METHODS` in
  `visualization.py` — the radio buttons and sliders pick it up automatically.

## How to test

```bash
uv run pytest tests/ -q          # all augmentation unit tests
uv run pytest tests/test_geometric.py -q
uv run flake8                    # lint (rules in .flake8)
```
