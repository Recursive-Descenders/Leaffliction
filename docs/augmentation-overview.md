# Augmentation Pipeline — Overview

**Context**: Leaffliction (42 RNCP). EDA findings from issue #4.  
**Spec**: [spec.md](spec.md) · **Decision graph**: [decision-graph.md](decision-graph.md)

---

## Why augmentation?

The dataset has a 6× class imbalance — the largest class holds 1 640 images while the smallest (`apple_rust`) has only 275. Training on raw counts teaches a model to favour the majority, not to classify. Augmentation fills the gap by generating synthetic images from existing sources.

---

## What EDA told us

Before writing a single line of augmentation code, EDA (issue #4) established four hard constraints:

| Finding | Implication for augmentation |
|---|---|
| **B1 — colour is signal** | Hue and saturation carry disease identity; they are low-freedom axes. |
| **B2 — shape is not signal** | Flip, rotate, shear, skew, crop are high-freedom axes; use literature defaults. |
| **B3 — features are scattered** | No fixed lesion location → large-area erasure outright rejected. |
| **B4 — all images are top-down** | No need to simulate viewing angles. |
| **C2 — known duplicates** | 7 duplicate pairs in `apple_healthy/`; dedup must run before anything else. |

These findings set the ceiling for what each transform is allowed to do.

---

## Pipeline design decisions

### K=2 random subset (#1)

Each output image gets exactly **2 transforms** drawn at random from the pool, each applied at its calibrated safe magnitude.

- 1 transform per output: too shallow — existing `Augmentation.py` already does this; the model never sees compound conditions (e.g. tilted *and* underexposed).
- Stack all N: B1 colour signal cannot survive 6-layer compounding.
- K=2 exposes the model to realistic combinations without destroying the colour signal.

### Calibration: literature + EDA heuristic (#2)

Start from RandAugment / Albumentations default ranges, then tighten per EDA group:

- **High freedom** (B2 — shape not signal): `flip / rotate / shear / skew / crop` → literature defaults.
- **Low freedom** (B1 — colour is signal): `hue / saturation` → tightest end of literature range (active post-S3).
- **Forbidden** (B3 — scattered features): large-area erasure rejected outright.

### Unified `aug PATH` CLI (#3′)

Three behavioural modes share one core, exposed through a single command that dispatches on path type:

```bash
# Mode A — single image → 5 K=2 outputs
uv run aug image.jpg --n 5

# Mode B — class folder → fill to target count
uv run aug apple_rust/ --target 1640

# Mode C — dataset root → balance all classes to max
uv run aug data/train/Apple/ --balance
```

The path type already encodes the mode; no need for three separate commands.

### Balance target: match max (#4)

`target = max(class_counts)` — minority classes are augmented up; the majority class is left untouched.

`apple_rust` needs 1 640 − 275 = **1 365 synthetic images** from 275 sources, roughly a 5× synthetic-to-real ratio.

### Source sampling: round-robin (#5)

`floor(need / N_sources)` copies per source; the first `need % N_sources` sources get one extra. Every source contributes as evenly as possible — no source is skipped entirely.

### Dedup pre-step (#6)

SHA-256 byte-hash dedup runs before any pipeline stage. Without it, duplicate sources produce duplicate augmented copies that inflate one source's effective weight.

### Pre-split input assumption (#7′)

The augmentor assumes the caller has already split train and val. It never touches val. No `split` CLI ships — users have their own preferred split tooling and we do not own that UX.

### Method pool: geometric-only MVP (#8)

`POOL = [flip, rotate, shear, skew, crop, radial_distortion]`

K=2 combinations: `C(6, 2) = 15`.

`color_jitter` is the natural next addition but is blocked on the S3 photometric stage. Without it, the model has no resilience to test-time lighting variation.

### Output filename (#9)

`<stem>_<Method1>_<Method2>_<i>.<ext>`

Example: `apple_rust_001_Rotate_Shear_0.jpg`

Methods are listed in call order; the index is per-source. Easier to debug than an opaque `_aug_0` suffix.

---

## Reproducibility

`--seed 42` is the default. Pass `--seed` to any mode:

```bash
uv run aug data/train/Apple/ --balance --seed 42
```

Omit `--seed` (or pass `None`) for stochastic runs.

---

## Next steps

### Planned upgrades (mapped, not yet shipped)

| # | What | Trigger | Upgrade path |
|---|---|---|---|
| **#8** | Add `color_jitter` to pool | When S3 (`apply_color_jitter`) ships | Append to `POOL`; combinations grow from 15 → 21 |
| **#2** | Re-calibrate magnitudes | Training reveals overfit / underfit tied to augmentation strength | Add sampled K=2 human stress test on hue / saturation axes |

### On-demand (open if needed)

- **Perceptual dedup (#6b)** — reopen if exact hash misses visually identical images.
- **Small-area erasure** — area < 10% was noted as theoretically safe, but B3 argues against it until training evidence says otherwise.
