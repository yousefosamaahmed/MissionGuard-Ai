# Dataset Card — OPSSAT-AD v2

## Dataset identity

- **Name:** OPSSAT-AD v2
- **Mission:** ESA OPS-SAT CubeSat
- **DOI:** `10.5281/zenodo.15108715`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Project use:** training, validation, official held-out testing, real telemetry visualization, and CSV upload demonstrations

## Packaged files

- `data/opssat/raw/dataset.csv` — 2,123 engineered segment rows with 23 columns.
- `data/opssat/raw/segments.csv` — 303,493 original telemetry samples across 2,123 segments.
- `data/opssat/processed/train_features.csv` — internal model-fit subset from the official training partition.
- `data/opssat/processed/validation_features.csv` — internal threshold-selection subset from the official training partition.
- `data/opssat/processed/test_features.csv` — untouched official test partition.
- `data/opssat/upload_samples/` — real held-out test examples for website upload testing.

## Telemetry channels

The dataset contains nine documented channel identifiers:

- `CADC0872`, `CADC0873`, `CADC0874` — three magnetometer axes.
- `CADC0884`, `CADC0886`, `CADC0888`, `CADC0890`, `CADC0892`, `CADC0894` — six photodiode channels.

MissionGuard preserves the source channel identifiers and does not rename undocumented signals as battery, engine, pressure, vibration, or radiation measurements.

## Labels and split

- `anomaly = 0` means normal/nominal segment.
- `anomaly = 1` means anomalous segment.
- `train = 1` belongs to the official training partition.
- `train = 0` belongs to the official held-out test partition.

The official split is preserved. Test labels are not used for training or threshold selection.

## Processing applied by MissionGuard

1. Validate required columns and numeric values.
2. Remove exact duplicate rows and reject files with excessive invalid data.
3. Extract the published segment-level engineered statistics from raw uploads.
4. Fit preprocessing and models on official-training data only.
5. Select decision thresholds on an internal validation subset of official training.
6. Evaluate once on the untouched official test split.

## Intended use

- Reproducible anomaly-detection research.
- Educational spacecraft-telemetry demonstrations.
- Human-in-the-loop decision-support prototypes.

## Known limitations

- The data covers nine selected channels, not the full spacecraft telemetry system.
- Labels are binary at segment level and do not provide hardware root cause or anomaly subtype.
- Distribution drift may reduce model reliability.
- A detected statistical anomaly is not proof of a hardware failure.
- The dataset and model are not certified for autonomous flight operations.

## Attribution

Bogdan Ruszczak, Krzysztof Kotowski, Jakub Nalepa, and David Evans, **OPSSAT-AD — anomaly detection dataset for satellite telemetry**, Zenodo, DOI: `10.5281/zenodo.15108715`.
