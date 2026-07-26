# Real OPSSAT-AD Upload Samples

All CSV files in this directory are copied from the **official held-out test split** in
`data/opssat/raw/segments.csv`. The `anomaly` column is retained only so the application can
score predictions after inference. It is never used as a model input.

| File | Official test segments | Purpose |
|---|---|---|
| `opssat_real_normal.csv` | 1964 | One labeled normal magnetometer segment |
| `opssat_real_anomaly.csv` | 1967 | One labeled anomalous magnetometer segment |
| `opssat_real_mixed.csv` | 1964, 1967, 1968 | Chronological normal → anomaly → normal/recovery demonstration |
| `opssat_real_magnetometer_anomalies.csv` | 13, 444, 239 | Anomalous examples from the three documented magnetometer channels |
| `opssat_real_photodiode_anomalies.csv` | 1068, 235, 1793 | Anomalous examples from three documented photodiode channels |

The dataset does not provide hardware root-cause or anomaly-subtype labels. These files are
therefore described only by their documented channel and binary normal/anomaly ground truth.
