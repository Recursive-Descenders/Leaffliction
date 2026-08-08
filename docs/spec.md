# Spec — Augmentor Pipeline

**Converged from**: [decision-graph.md](decision-graph.md) (11 nodes, all resolved).
**Date**: 2026-06-09
**Context**: Leaffliction (42 RNCP). EDA findings from issue #4 (6× class imbalance, B1 color = signal, B2 shape not signal, B3 features scattered, B4 top-down only, C2 known duplicates).

---

## ⚠️ Unresolved Premises

Two adopted decisions carry **planned re-assumption flags** — they ship as MVP defaults but the upgrade path is already mapped.

| Node | Flag | Trigger to revisit | Upgrade path |
|---|---|---|---|
| **#2 Calibration** | revisit-candidate | Model training reveals overfit / underfit tied to augmentation magnitudes | `d` (literature + EDA heuristic) → `e` (add K=2 sampled human stress test on hue / saturation / erasure axes) |
| **#8 Method pool** | revisit-planned | When S3 (`apply_color_jitter`) ships | `A` (geometric-only, 6 methods) → `B` (append `color_jitter`, pool grows to 7) |

Neither blocks MVP shipping; both are tracked openly so the team knows where the seams are.

---

## Adopted Decisions (in dependency order)

### Architecture core

#### #1 — Composition strategy: K=2 random subset (RandAugment-lite)
Each output image = 2 transforms drawn at random from the pool, each applied at its calibrated safe magnitude.
- Rejected `a` (1 transform per output) — no exposure to compound real-world conditions.
- Rejected `c` (stack all N) — B1 color signal cannot survive 6-layer compounding.
- Rejected `d` (mixed counts) — complexity without clear win.

#### #2 — Calibration strategy: literature + EDA heuristic ⚠️ revisit-candidate
Take RandAugment / Albumentations defaults; tighten per EDA group:
- High freedom (B2 shape not signal): `flip / rotate / shear / skew / crop` → literature defaults.
- Low freedom (B1 color = signal): `hue / saturation` → tightest end (post-S3 ship).
- Forbidden (B3 features scattered): large-area erasure rejected outright.

Rejected: `a` manual-only (slow, subjective), `b` full K=2 stress test (O(N²) overkill pre-training), `c` classifier-feedback (chicken-and-egg + bias risk), `e` recommended hybrid (deferred for MVP simplicity).

#### #3 / #3' — Scope: A+B+C behavior, **unified `aug PATH` CLI**
The three behavioural modes (A single-image / B class-folder / C dataset-root) share one Augmentor core and now sit behind a single command that dispatches on path type:
- `aug image.jpg --n 5` — image file → mode A, 5 K=2 outputs.
- `aug apple_rust/ --target 1640` — class folder → mode B, fill to target.
- `aug data/train/Apple/ --balance` — dataset root → mode C, balance all classes to max.

#3 originally proposed three named CLI entries (`aug` / `aug-class` / `aug-dataset`); #3' supersedes that by collapsing them — the path type already encodes the mode unambiguously and end-users already know whether they hold an image or a directory.

### Dataset-level decisions (mode C)

#### #4 — Balance target: match max class count
`target = max(class_counts)`. Minorities augmented up; majority untouched.
- `apple_rust` needs `1640 - 275 = 1365` synthetic copies from 275 sources → ~5× synthetic-to-real ratio.
- Rejected `b` configurable N (requires throwing real data when N < max), `c` per-class multiplier (doesn't fix relative imbalance), `d` opt-in majority augmentation (no clear benefit, complicates baseline).

#### #5 — Source sampling: round-robin + seeded RNG
Deterministic floor + remainder: `floor(need / N_sources)` per source, first `need % N_sources` sources produce one extra.
- Rejected `α` random-with-replacement (variance leaves some sources unused), `γ` shuffle cycles (no value over β), `δ` stratified by attribute (no attribute labels).

#### #6 — Dedup pre-step: YES (exact file-hash)
SHA-256 / MD5 byte-hash dedup runs before any pipeline stage. EDA C2 found 7 pairs in `apple_healthy/`; without dedup their augmented copies inherit duplicate weight.
- Sub-detail `#6b` (perceptual / near-dup hash) deferred — reopen if exact-hash misses real near-dups.

#### #7 / #7' — Pre-split input assumption (no `split` CLI)
The augmentor assumes input is already split by the user — they pass the train side (image / class folder / dataset root) and val never enters the pipeline. The leak hazard #7 originally guarded against (augmented copies spanning train and val) is now a usage-error concern rather than a tool design concern, traded for a smaller surface area.

#7 originally shipped a `split` CLI as a separate step; #7' supersedes that by dropping the tool entirely. End-users already have a preferred way to split a dataset; we don't own that UX.

### Pool and interface details

#### #8 — Method pool: geometric-only (MVP) ⚠️ revisit-planned
`POOL = [flip, rotate, shear, skew, crop, radial_distortion]`. Ships now to unblock Part 4 teammates. `K=2` combinations = `C(6,2) = 15`.
- Rejected `B` (6 + color_jitter) — blocked on S3; queued as upgrade.
- Rejected `C` (6 + color_jitter + erasure) — erasure unsafe under B3 (scattered features).
- Rejected `D` (configurable pool flag) — over-engineered.

#### #9 — Output filename: `<stem>_<Method1>_<Method2>_<i>.<ext>`
Records the two methods in call-order + per-source index.
- Example: `apple_rust_001_Rotate_Shear_0.jpg`.
- Rejected index-only `<stem>_aug_<i>` (harder to debug) and subject `<stem>_<Method>` (only fits K=1).

#### #10 — Interface: class-based `Augmentor`
`Augmentor(pool, seed).apply(image) → image`. Holds RNG + pool + magnitude ranges.
- Rejected pure functional API (param-heavy at call sites).

#### #11 — Seed: CLI `--seed 42` default
Default seeded for reproducible MVP runs; `None` opts into stochastic.

---

## Implementation Skeleton

Actual layout shipped:

```
src/
  Augmentation.py      # single `aug PATH` CLI — dispatches by path type (#3' + #11)
  augmentation/
    geometric.py       # existing 6 methods (flip / rotate / shear / skew / crop / radial_distortion)
    registry.py        # POOL of Method dataclasses (#8) — magnitudes from #2
    core.py            # Augmentor class (#10) — pool + seeded RNG + K=2 apply (#1)
    preprocess.py      # exact file-hash dedup + image listing (#6)
    util.py            # build_aug_output_path (#9) + load/save helpers
    visualization.py   # interactive matplotlib viewer (orthogonal, dev-only)
tests/
  test_geometric.py
  test_augmentor.py    # POOL composition, K=2 reproducibility, filename, dedup
```

Per #7' there is no separate `split` CLI — users are expected to pre-split their dataset before invoking `aug`.

### Mode C control flow (dataset root)

```
1. For each class folder under data_dir:                              [#3' mode C]
     paths      = list_images(class_folder)
     kept, dup  = dedup_images(paths)                                 [#6]
     need       = target - len(kept)                                  [#4]
     if need <= 0: continue
     per_source = need // len(kept)
     remainder  = need % len(kept)
     for i, source in enumerate(kept):                                [#5 round-robin]
         copies = per_source + (1 if i < remainder else 0)
         for k in range(copies):
             image     = load(source)
             augmented = augmentor.apply(image)                       [#1 + #10]
             out_name  = build_aug_output_path(source,                [#9]
                            augmentor.last_methods, k)
             save(augmented, class_folder / out_name)
```

---

## Final Panorama

```mermaid
graph TD
  subgraph Architecture
    N1["#1 composition<br/>✅ K=2 random subset"]:::active
    N2["#2 calibration<br/>✅ literature + EDA<br/>⚠️ revisit-candidate"]:::active
    N3["#3 scope<br/>✅ A+B+C share core"]:::active
  end

  subgraph Dataset["Dataset mode C"]
    N4["#4 balance target<br/>✅ match max class"]:::active
    N5["#5 source sampling<br/>✅ round-robin + seed"]:::active
    N6["#6 dedup pre-step<br/>✅ YES, exact hash"]:::active
    N7["#7 split timing<br/>✅ separate split CLI"]:::active
  end

  subgraph Pool["Pool + interface"]
    N8["#8 method pool<br/>✅ geometric-only<br/>⚠️ revisit-planned"]:::active
    N9["#9 filename<br/>✅ stem_M1_M2_i"]:::active
    N10["#10 interface<br/>✅ Augmentor class"]:::active
    N11["#11 seed<br/>✅ --seed 42"]:::active
  end

  N1 --> N2
  N1 --> N3
  N3 --> N4
  N4 --> N5
  N3 --> N6
  N3 --> N7
  N1 --> N8
  N1 --> N9
  N8 --> N9
  N1 --> N10
  N5 --> N11

  classDef active fill:#1f6feb,color:#fff
```

---

## Handoff

The spec is ready to flow downstream:

- **`/to-issues`** — break this spec into independently-grabbable issues for teammates (recommended for ship-MVP-fast).
- **`/to-prd`** — promote to a single PRD if you want a higher-level document for stakeholder alignment.
