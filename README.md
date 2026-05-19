# 🍁 Leaffliction

**Leaf image classification** — analysis, augmentation, transforms, then train / predict (with validation as per subject).

## Environment

We use **uv** as the package manager for Python. Python **3.12**; code under **`src/`**. **`uv.lock`** locks deps. **`.python-version`** is `3.12`. The dataset is **not** in the repo (see **`.gitignore`**); keep it locally.

### Install `uv` (once per machine)

``curl -LsSf https://astral.sh/uv/install.sh | sh`` or ``pipx install uv``

### Dependencies and venv

```bash
uv sync
```

Creates **`.venv`**, applies **`uv.lock`**, and installs deps (including **flake8**).


### `uv run <name>`

**Console scripts** are defined in **`[project.scripts]`** in **`pyproject.toml`** (short names for typing; the comment in that file says you can rename for subject/eval if needed). After **`uv sync`**:

| Command     | `src` module     |
|-------------|------------------|
| `uv run dist`  | `Distribution.py`  |
| `uv run aug`   | `Augmentation.py`  |
| `uv run xfm`   | `Transformation.py` |
| `uv run tr`    | `train.py`         |
| `uv run pr`    | `predict.py`       |

### Code layout

Same pattern for every module:

1. **`argparse`** for flags — keep each top-level `*.py` entry file thin (parse args, then call your code).
2. **Entry + folder** — logic lives in a package under `src/`, not in the entry file (e.g. `Transformation.py` → `transformation/`).
3. **Outputs** — if you save to the repo, use `outputs/<module>/` (`outputs/distribution/`, `outputs/augmentation/`, `outputs/transformation/`, …). Follow the subject when it says otherwise.

### flake8

Per subject: **`pip install flake8`**-style check — here it is a dev dep, so use it after `uv sync`.

```bash
uv run flake8
```

## Pull requests

**PRs** run [**`.github/workflows/lint.yml`**](.github/workflows/lint.yml): **`uv sync --frozen`** then **`uv run flake8 .`** (same rules as **`.flake8`** locally). If it fails, fix and push.

**Before you open a PR**, run the same check yourself:

```bash
uv sync
uv run flake8 
```