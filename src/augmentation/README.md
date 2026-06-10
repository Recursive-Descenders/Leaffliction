# Augmentation

Image augmentation module for the **Leaffliction** 42 project.

Design decisions and rationale live in [`docs/decision-graph.md`](../../docs/decision-graph.md);
the user-facing contract is in [`docs/spec.md`](../../docs/spec.md).

## Module layout

| File | Role |
| --- | --- |
| `geometric.py` | The six geometric transforms. Pure functions on a BGR `np.ndarray`. |
| `registry.py` | `POOL` — the `Method` entries the Augmentor samples from, with calibrated parameter ranges (decision #2). |
| `core.py` | `Augmentor` — draws K=2 distinct methods from the pool per call, samples magnitudes, applies in call order (decisions #1, #10, #11). |
| `preprocess.py` | `list_images`, `dedup_images` (byte-identical SHA-256 dedup, decision #6). |
| `util.py` | `load_image`, `save_image`, `build_aug_output_path` (naming rule, decision #9). |
| `visualization.py` | Matplotlib preview viewer — radio buttons + per-method sliders. |

The CLI entry is `src/Augmentation.py`, exposed as `aug` in `pyproject.toml`.

## The pool

Six geometric methods, all implemented and live in `POOL`:

| Method | Parameters (sampling range) |
| --- | --- |
| `Flip` | — |
| `Rotate` | `angle ∈ [-25°, 25°]` |
| `Shear` | `kx, ky ∈ [-0.2, 0.2]` |
| `Skew` | `skx, sky ∈ [-0.3, 0.3]` |
| `Crop` | `scale ∈ [0.7, 1.0]` |
| `Distortion` | `k ∈ [-0.2, 0.2]` (radial) |

Ranges are the "high freedom" band per decision #2 (B2 shape is not the
classification signal). Color jitter (`hue`, `saturation`) is the planned
S3 addition once color preprocessing lands — see decision #8.

## Composition: K=2

Every CLI call hands work to an `Augmentor`. Each `Augmentor.apply(image)`:

1. Draws **two distinct** methods from `POOL` without replacement.
2. Samples a magnitude for each parameter uniformly from its range.
3. Applies the two methods in call order.
4. Records the method names on `last_methods` so the IO layer can encode
   them into the output filename.

The Augmentor reseeds NumPy's global RNG from its own seeded RNG before
every composition, so methods that still use `np.random.*` internally
(notably `apply_crop`'s random offset) stay reproducible under `--seed`.

## CLI: one entry, three modes

```bash
# Mode A — single image: write N K=2 variants under --dst.
uv run aug "data/raw/Apple/apple_healthy/image (1).JPG" --n 6

# Mode B — class folder: fill the folder to --target images.
uv run aug data/train/Apple/apple_rust --target 1640

# Mode C — dataset root: balance every class to the largest count.
uv run aug data/train/Apple --balance
```

Mode is dispatched on the input path:

- File → mode A.
- Directory that **directly** contains image files → mode B (class folder).
  Requires `--target N`.
- Directory whose children are class sub-folders → mode C (dataset root).
  Pass `--balance` to auto-target the max class count, or `--target N` for
  an explicit ceiling.

Modes B and C dedup byte-identical sources first, then round-robin over
the kept images so every source contributes evenly to the new totals.

### Flags

| Flag | Default | Mode | Meaning |
| --- | --- | --- | --- |
| `--dst` / `-d` | `data/augmented_directory` | all | Output root. Folder modes write under `<dst>/<class_name>/`; source folders are never mutated. |
| `--n` / `-n` | 6 | A | Number of K=2 variants to produce. |
| `--target` / `-t` | 0 | B, C | Target image count per class. 0 = auto-detect (C only). |
| `--balance` / `-b` | off | C | Auto-target the largest class count. |
| `--seed` / `-s` | 42 | all | RNG seed. Negative = stochastic. |

### Output naming

`build_aug_output_path` emits

```
<stem>_<Method1>_<Method2>_<i><suffix>
```

e.g. `image (1)_Flip_Crop_0.JPG`. The methods are listed in the order
they were applied; `i` is the per-source counter inside a single run.

`build_output_path` (the older single-method form,
`<stem>_<Method><suffix>`) is still exported for callers that want it
(e.g. the viewer's single-method snapshots).

## Interactive visualization

```bash
uv run python -m augmentation.visualization "data/raw/Apple_healthy/image (1).JPG"
# no argument → uses a synthetic test pattern from util.make_test_image()
uv run python -m augmentation.visualization
```

Original on the left, augmented on the right. Radio buttons switch
between methods; sliders are rebuilt per method from the viewer's local
`METHODS` table (separate from `registry.POOL` — the viewer exposes
wider, exploratory ranges than the calibrated sampling ranges).

Notes:

- Needs a GUI backend — run from a desktop session, not a headless shell.
- Preview only; nothing is written. Use `uv run aug ...` to save.
- Methods that raise `NotImplementedError` show a red placeholder instead
  of sliders. None of the current pool methods do, but the guard stays so
  new pool entries can land before their implementation does.

## Tests

```bash
uv run pytest tests/ -q                  # all augmentation unit tests
uv run pytest tests/test_geometric.py -q # geometric transforms only
uv run pytest tests/test_augmentor.py -q # Augmentor composition + seeding
uv run flake8                            # lint (rules in .flake8)
```
