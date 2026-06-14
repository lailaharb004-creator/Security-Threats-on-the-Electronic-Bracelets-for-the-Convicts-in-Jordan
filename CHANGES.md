# Project Changes & Migration Notes

This document summarizes how the detection code was upgraded from the original
**Autonomous-Vehicle (AV) GPS** version to the current **human bracelet GPS** version,
and what exactly was changed in the notebook and scripts.

---

## 1. Summary of what changed

- **Dataset:** switched from a borrowed car/AV GPS dataset to a realistic **human bracelet**
  dataset (per-second readings, Jordan coordinates) with **labeled spoofing attacks injected**
  by a custom attack generator.
- **Features:** grew from **13** satellite/velocity features to **33** movement-aware features
  (position, speed consistency, direction consistency, signal quality, rolling windows).
- **Models:** the 3-model soft-voting ensemble changed from
  *RF + Neural Net + HistGradientBoosting* to *RF + Neural Net + **Extra Trees***.
- **Validation:** added a **time-based split** and a **blind test** on unseen data
  (much more honest than a single shuffled split).
- **Repo:** files renamed (no spaces/typos) and organized into
  `report/`, `notebook/`, `code/`, `data/`, `outputs/`.

> Note: only the notebook logic and the file-path lines in the scripts were edited.
> Rawan's actual detection algorithms were **not** modified.

---

## 2. Raw input columns

| Old (AV dataset) | New (human bracelet dataset) |
|---|---|
| Satellite Count, Satellite Locks | satellites_in_view, satellites_used |
| Velocity (m/s) | velocity |
| Heading (deg), GPS Course | course |
| — *(no position)* | **latitude, longitude** |
| — *(no signal quality)* | **hdop** |
| — *(no time)* | **gps_date, gps_time** (+ session_id) |

The big additions are **GPS coordinates**, **HDOP** (signal quality), and **timestamps**,
which make real movement analysis possible.

---

## 3. Features: OLD vs NEW

### OLD — 13 features
```
sat_count, sat_locks, sat_ratio, sat_discrepancy
velocity, velocity_diff
heading_abs_diff
+ rolling mean/std (window 5) for: velocity, sat_count, sat_ratio
```

### NEW — 33 features
```
Satellite/signal:  sat_count, sat_locks, sat_ratio, sat_discrepancy,
                   hdop, hdop_diff
Movement/speed:    velocity, velocity_diff, acceleration,
                   distance_m, coord_speed, speed_residual
Direction:         course_filled, course_change, course_bearing_diff
Motion flags:      is_stationary, is_fast_human
+ rolling mean/std (window 5) for 8 of the above:
   velocity, coord_speed, speed_residual, sat_ratio,
   sat_discrepancy, hdop, course_change, course_bearing_diff
```

### The genuinely new ideas
| New feature | What it catches |
|---|---|
| `distance_m` | real distance moved between two GPS points (haversine) |
| `coord_speed` | speed implied by the coordinates vs. time |
| **`speed_residual`** | mismatch between reported `velocity` and `coord_speed` — a classic spoofing tell |
| **`course_bearing_diff`** | reported heading vs. actual direction of travel — another spoofing tell |
| `hdop`, `hdop_diff` | signal-quality changes (turned out to be the **strongest** indicators) |
| `is_stationary`, `is_fast_human` | flags for human-realistic motion |
| per-`session_id` grouping | features computed per track, not mixed across people |

**Why it matters:** the old features only asked *"how many satellites / how fast?"*.
The new ones ask *"does the movement physically make sense?"* by cross-checking position,
speed, direction, and signal quality — i.e. cross-feature forensic analysis.

---

## 4. Models: OLD vs NEW

Both use a **soft-voting ensemble of 3 models**, but the lineup changed.

| | OLD notebook | NEW notebook (Rawan's) |
|---|---|---|
| Model 1 | Random Forest — 300 trees, depth 15 | Random Forest — 100 trees, depth 10 |
| Model 2 | Neural Network (MLP) — layers (100, 50), 500 iters | Neural Network (MLP) — layer (50), 80 iters |
| Model 3 | **HistGradientBoosting** | **Extra Trees** — 150 trees, depth 12 |
| Combiner | Soft Voting (weights 1·1·1) | Soft Voting (weights 1·1·1) |

**Key swap:** HistGradientBoosting → **Extra Trees**, with lighter/faster models.

### Training & testing
| | OLD | NEW |
|---|---|---|
| Train/test split | 80 / 20 | 75 / 25 |
| Cross-validation | 5-fold CV | — |
| Time-based split | ❌ | ✅ (train on earlier data, test on later) |
| Blind test on unseen file | ❌ | ✅ (predict on a no-label file, then score it) |
| Per-attack-type scoring | ❌ | ✅ (accuracy per attack type) |

---

## 5. Results (current version)

| Evaluation | Accuracy |
|------------|----------|
| Ensemble – random split | ~99% |
| Random Forest (alone) | ~99.5% |
| Blind test (unseen data) | ~99% |

Precision, recall, and F1 are all high (≈98%+).

---

## 6. What was edited, file by file

### `notebook/fix.ipynb`
- Imports updated (added `ExtraTreesClassifier`, etc.).
- Config cell: new dataset paths, `OUTPUT_DIR`, `RANDOM_STATE`, `TEST_SIZE`, `ROLLING_WINDOW`.
- Replaced the whole class with `HumanGPSDetector` (new features + new models + time-based split).
- Added 3 cells: blind-test prediction + scoring against true labels + a markdown explainer.
- Fixed the no-label filename to match the renamed dataset.

### `code/` scripts — paths only (logic untouched)
| File | Change |
|------|--------|
| `detection_code.py` | `DATASET_FILE` → `data/...`, `OUTPUT_DIR` → `outputs` |
| `predict_unlabeled.py` | `UNLABELED_FILE` → `data/...`, model paths → `outputs/...` |
| `compare.py` | `true_file` → `data/...`, `pred_file` → `outputs/...` |
| `generate_spoofing_attack.py` | input/output paths → `data/...` and `outputs/...` |

### Not changed
Datasets, the trained model (`.pkl`), the plots, and Rawan's algorithms — only renamed/moved.

---

## 7. ⚠️ Report vs. code mismatch (to fix before submission)

The written report (`report/…pdf`) lists a **different** model set than the code:

- **Report PDF says:** Decision Tree, Random Forest, Gradient Boosting, Logistic Regression
- **Notebook now uses:** Random Forest, Extra Trees, Neural Network

The report's **features** section is also based on the old AV dataset. Both the model list and
the feature list in the report should be updated to match the current code.
