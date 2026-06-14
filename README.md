# Security Threats on Electronic Bracelets for the Convicts in Jordan

### GPS Spoofing Detection for Electronic Monitoring Bracelets

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Cybersecurity](https://img.shields.io/badge/Cybersecurity-GPS%20Spoofing-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

A graduation project (B.Sc. Cyber Security, The Hashemite University) that detects **GPS spoofing
attacks** on the electronic ankle bracelets used to monitor convicts. The system uses machine
learning to tell **legitimate** GPS records apart from **spoofed (fake)** ones by analyzing
movement, satellite, and signal-quality behavior.

---

## The Problem

Electronic Monitoring (EM) bracelets trust the GPS data they receive without verifying it. An
attacker can therefore forge that data and make an offender *appear* inside an allowed zone while
they are actually somewhere else. The main attack vectors are:

- **GPS signal spoofing** – broadcasting fake satellite signals
- **Replay attacks** – re-sending old, valid GPS data at a later time
- **Timestamp / sequence manipulation**
- **Satellite & sensor data fabrication**

## Our Approach

1. **Build a realistic dataset.** No public dataset for human ankle bracelets exists, so we started
   from realistic normal human GPS tracks (per-second readings, Jordan coordinates) and used a
   custom **attack generator** to inject labeled spoofing attacks into them
   (gradual drag, freeze, replay, fake walking, jump-and-recover).
2. **Engineer forensic features.** From the raw GPS stream we compute per-session movement and
   consistency features (see below).
3. **Train an ensemble model** and validate it honestly with a random split, a time-based split,
   and a **blind test** on data the model never saw.

> An earlier version of the project used a borrowed Autonomous-Vehicle GPS dataset; the current
> version (in `data/` and `code/`) uses the realistic human-bracelet dataset.
> See [`CHANGES.md`](CHANGES.md) for a full before/after of the features, models, and structure.

---

## Repository Structure

```text
.
├── README.md
├── CHANGES.md                       # AV→human migration notes (features, models, structure)
├── report/
│   └── Project1_Electronic_Bracelet_Attack_detection.pdf   # Full written report
├── notebook/
│   └── fix.ipynb                  # Google Colab notebook (end-to-end pipeline)
├── code/
│   ├── detection_code.py          # Train + evaluate the detector, save model & plots
│   ├── predict_unlabeled.py       # Load saved model, predict on unseen data (deployment)
│   ├── compare.py                 # Score blind predictions vs. true labels
│   └── generate_spoofing_attack.py# Inject realistic spoofing attacks into normal GPS data
├── data/
│   ├── gp_data_normal.csv               # Normal human GPS tracks (attack generator input)
│   ├── gps_data_spoofed_3.csv           # Labeled training data (normal + spoofed)
│   ├── gps_data_spoofed_3000.csv        # Labeled data with ground truth
│   ├── gps_data_spoofed_3000_no_label.csv# Same rows without labels (blind test)
│   └── gps_data_3000_record.csv
└── outputs/
    ├── models/                    # Saved ensemble model + preprocessing (.pkl)
    ├── plots/                     # Confusion matrices, feature importance, timelines, trajectory
    ├── human_gps_predictions_*.csv
    ├── unlabeled_predictions_*.csv
    └── human_model_report_*.txt   # Run reports
```

---

## Detection Features

Computed per `session_id` from the raw GPS stream:

| Group | Features |
|-------|----------|
| Satellite / signal | `sat_count`, `sat_locks`, `sat_ratio`, `sat_discrepancy`, `hdop`, `hdop_diff` |
| Movement / speed | `distance_m` (haversine), `coord_speed`, `speed_residual`, `velocity`, `velocity_diff`, `acceleration` |
| Direction | `course_filled`, `course_change`, `course_bearing_diff` |
| Motion flags | `is_stationary`, `is_fast_human` |
| Rolling (attack-window) | mean & std over a 5-step window for 8 of the above |

The strongest spoofing indicators were the **HDOP** (signal-quality) features and the rolling
satellite/course features.

---

## Machine Learning Models

| Model | Role |
|-------|------|
| Random Forest | Main classifier + feature importance |
| Extra Trees | Variance reduction |
| Neural Network (MLP) | Captures non-linear patterns |
| **Voting Classifier** | **Soft-voting ensemble of the three (final model)** |

---

## Results

| Evaluation | Accuracy |
|------------|----------|
| Ensemble – random split | ~99% |
| Random Forest (alone) | ~99.5% |
| Blind test (unseen data) | ~99% |

Precision, recall, and F1 are all high (≈98%+), with a low false-positive rate.

---

## How to Run

### Option A — Google Colab (recommended)

1. Open `notebook/fix.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload the three CSVs from `data/` to your Google Drive at the path set in the config cell.
3. Run all cells. The notebook trains the model, evaluates it, runs the blind test, and saves
   plots/results.

### Option B — Python scripts (run from the repo root)

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib

python code/detection_code.py        # train + evaluate, saves model into outputs/
python code/predict_unlabeled.py     # predict on the unlabeled data
python code/compare.py               # score those predictions vs. ground truth
```

---

## Technologies

Python · Pandas · NumPy · Scikit-Learn · Matplotlib · Seaborn

---

## Future Work

- Real-time GPS monitoring and live alerting
- NS-3 network simulation of bracelet ↔ monitoring server
- Validation on real Electronic Monitoring data from judicial authorities
- Attack **prevention** mechanisms (current scope is detection)

---

## Team

- Laila Ali Harb
- Saja Abdallah Alqawareeq
- Rawan Mohammad Abed
- Maram Husam Zaza
- Hanan Hazim Alabsa

**Supervisor:** Dr. Musab Alghadi

**The Hashemite University** — Faculty of Prince Al-Hussein Bin Abdullah II for Information Technology,
Department of Cyber Security.

---

## License

Developed for academic and research purposes as part of the B.Sc. Cyber Security degree at
The Hashemite University.
