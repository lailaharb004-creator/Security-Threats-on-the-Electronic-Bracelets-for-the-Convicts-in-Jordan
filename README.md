# GPS Spoofing Detection for Electronic Monitoring Bracelets

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Cybersecurity](https://img.shields.io/badge/Cybersecurity-GPS%20Spoofing-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Overview

Electronic Monitoring (EM) bracelets are widely used by judicial institutions to track offenders through GPS-based location monitoring. However, these systems are vulnerable to GPS spoofing attacks, where forged GPS signals manipulate the reported location of the monitored individual.

This project presents a Machine Learning-based detection framework capable of identifying GPS spoofing attempts by analyzing GPS behavioral patterns, satellite information, movement characteristics, and cross-feature inconsistencies.

The proposed solution improves the integrity, reliability, and security of Electronic Monitoring systems by automatically distinguishing legitimate GPS records from manipulated ones.

---

## Problem Statement

Current Electronic Monitoring systems often trust raw GPS data without validating its authenticity.

This creates several security risks:

- GPS Signal Spoofing
- Replay Attacks
- Timestamp Manipulation
- Sensor Data Fabrication
- False Location Reporting

As a result, offenders may appear to comply with movement restrictions while actually violating them.

---

## Project Objectives

- Analyze cybersecurity threats targeting GPS-based Electronic Monitoring systems.
- Study the characteristics of GPS spoofing attacks.
- Identify forensic indicators of manipulated GPS data.
- Develop a Machine Learning detection model.
- Improve trust and reliability in offender monitoring systems.
- Generate automated alerts for suspicious GPS behavior.

---

## System Architecture

The detection system follows a complete Machine Learning pipeline:

```text
GPS Dataset
     │
     ▼
Data Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
Model Training
     │
     ▼
Ensemble Classification
     │
     ▼
Spoofing Detection
     │
     ▼
Reporting & Visualization
```

---

## Features

### Data Processing

- Missing Value Handling
- Feature Scaling
- Label Normalization
- GPS Feature Extraction

### GPS Security Analysis

- Satellite Count Analysis
- Satellite Lock Analysis
- Velocity Monitoring
- Heading Consistency Verification
- GPS Behavior Validation

### Machine Learning Detection

- Binary Classification
- Ensemble Learning
- Confidence Scoring
- Anomaly Detection

### Reporting

- Confusion Matrix
- Feature Importance Analysis
- Model Comparison
- Prediction Distribution
- Final Detection Reports

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## Machine Learning Models

| Model | Purpose |
|---------|----------|
| Decision Tree | Interpretable classification |
| Random Forest | Improved accuracy and feature importance |
| Gradient Boosting | Detection of complex attack patterns |
| Logistic Regression | Baseline classification |
| Voting Classifier | Ensemble-based final prediction |

The final system uses a Soft Voting Ensemble Model that combines all classifiers to maximize detection performance and reduce model-specific bias.

---

## Dataset

Due to the lack of publicly available datasets related to Electronic Monitoring bracelets, the project utilizes a GPS Spoofing Dataset originally developed for Autonomous Vehicle environments.

The dataset contains:

- Legitimate GPS Records
- Spoofed GPS Records
- Satellite Information
- Velocity Measurements
- Navigation Parameters

---

## Detection Features

The model analyzes several GPS attributes, including:

| Feature | Description |
|----------|------------|
| Satellite Count | Number of visible satellites |
| Satellite Locks | Number of locked satellites |
| Velocity | Device movement speed |
| GPS Speed | GPS-calculated speed |
| Heading Difference | Difference between actual and reported direction |
| Time Interval | Time between GPS records |
| Altitude Change | Variation in altitude |

---

## Results

### Performance Summary

- Detection Accuracy: **98%**
- High Precision Classification
- Low False Positive Rate
- Strong Class Separation

### Key Findings

- Satellite Count was among the strongest spoofing indicators.
- Satellite Locks significantly influenced model decisions.
- Ensemble Learning outperformed individual classifiers.
- GPS spoofing can be effectively detected through behavioral analysis and feature correlation.

---

## Generated Outputs

```text
Project Outputs
│
├── confusion_matrix.png
├── feature_importance.png
├── anomaly_scores.png
├── model_comparison.png
├── prediction_distribution.png
├── predictions.csv
└── final_report.txt
```

---

## Future Work

Future enhancements include:

- Real-Time GPS Monitoring
- IoT Device Integration
- NS-3 Network Simulation
- Synthetic Attack Generation
- Live Alerting System
- Validation Using Real Electronic Monitoring Data
- Attack Prevention Mechanisms

---

## Research Impact

This project contributes to the growing field of IoT and Cybersecurity by demonstrating how Machine Learning can strengthen the security of GPS-based offender monitoring systems and improve resistance against location-manipulation attacks.

---

## Team Members

- Laila Ali Harb
- Saja Abdallah Alqawareeq
- Rawan Mohammad Abed
- Maram Husam Zaza
- Hanan Hazim Alabsa

### Supervisor

Dr. Musab Alghadi

---

## Institution

**The Hashemite University**  
Faculty of Prince Al-Hussein Bin Abdullah II for Information Technology  
Department of Cyber Security

---

## License

This project was developed for academic and research purposes as part of the Bachelor of Cyber Security degree requirements at The Hashemite University.
