# Model Card — MissionGuard OPSSAT Hybrid v4

## Purpose

Human-in-the-loop, segment-level anomaly detection for real OPSSAT-AD telemetry.

## Components

- Isolation Forest trained only on nominal official-training segments.
- Random Forest trained on labeled official-training segments.
- Hybrid anomaly score: 42% calibrated Isolation Forest evidence + 58% uncalibrated supervised score.
- Packaged preprocessing and model artifacts loaded at application startup.

## Data separation

- Official training split: model fitting.
- Internal stratified validation subset: threshold selection.
- Official test split: final evaluation only.
- Ground-truth labels are excluded from inference features.

## Outputs

- Isolation score
- Supervised anomaly score
- Hybrid anomaly score
- Normal/Watch/Warning/Critical risk band
- Decision margin from the saved hybrid threshold
- Local engineered-feature deviation evidence
- Data compatibility and drift assessment

## Evaluation

The project reports segment-level and event-level metrics, including precision, recall, F1, MCC, PR-AUC, ROC-AUC, false alarms, missed anomalies, and an event ledger.

## Limitations

The model detects statistical anomalies and does not prove a hardware fault or causal root cause. The supervised score is not a calibrated failure probability. Distribution drift may reduce reliability. The system is not certified flight software.
