# MissionGuard AI

## Real ESA OPS-SAT Telemetry Anomaly Detection

MissionGuard AI is an explainable spacecraft-telemetry decision-support application built on **real telemetry acquired aboard ESA's OPS-SAT CubeSat**. It preserves the official OPSSAT-AD train/test split, evaluates on unseen test segments, visualizes original telemetry, validates uploaded CSV files, measures event-level detection, and monitors distribution drift.

> **One-line pitch:** MissionGuard AI detects anomalous OPS-SAT telemetry segments, explains the strongest engineered evidence, checks whether incoming data still resembles training conditions, and keeps human operators in the decision loop.

---

## Live Demo

MissionGuard AI is currently deployed and available at:

[https://missionguardplatform.duckdns.org](https://missionguardplatform.duckdns.org)

The public deployment runs on AWS Lightsail using Docker Compose, PostgreSQL, pgAdmin, Nginx, and HTTPS.

The Streamlit application, PostgreSQL, and pgAdmin ports are bound to localhost and are not exposed directly to the public internet.

---

## Challenge UI/UX

The application now opens on a cinematic **Launchpad** that introduces MissionGuard AI before the user enters mission control. Every major capability has its own numbered workspace, and the new **Team & Contact** page presents the builders and their roles professionally. The interface remains responsive and supports both Dark and Light appearance modes.

## Dataset

**OPSSAT-AD v2** contains telemetry acquired aboard OPS-SAT, a CubeSat mission operated by the European Space Agency.

- Dataset DOI: `10.5281/zenodo.15108715`
- License: **CC BY 4.0**
- Raw samples: 303,493
- Engineered segments: 2,123
- Channels: 9
- Official training segments: 1,594
- Official test segments: 529

The application preserves original channel identifiers. It does not rename undocumented signals as battery, engine, pressure, vibration, or radiation sensors.

The full data card is available at:

```text
data/DATASET_CARD.md
```

---

## Model Pipeline

MissionGuard packages pre-trained artifacts so the Streamlit website **does not retrain models on every launch**.

1. **Isolation Forest**
   - Trained only on nominal segments from the official training split.
   - Detects deviation from the learned nominal envelope.

2. **Supervised Random Forest**
   - Trained on labeled normal and anomalous official-training segments.
   - Produces an uncalibrated anomaly score.

3. **Hybrid anomaly score**
   - 42% calibrated Isolation Forest evidence.
   - 58% supervised Random Forest score.
   - Threshold selected only on an internal validation subset from official training.

Ground-truth labels are excluded from all inference features.

### Packaged artifacts

```text
models/opssat_model.joblib
models/opssat_isolation_bundle.joblib
models/opssat_supervised_bundle.joblib
models/opssat_feature_columns.json
models/opssat_metadata.json
models/opssat_metrics.csv
```

---

## Official Test Results

| Model | Precision | Recall | F1 | MCC | PR-AUC | ROC-AUC | False Alarms / 1000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.4444 | 0.4602 | 0.4522 | 0.3001 | 0.4590 | 0.7286 | 156.25 |
| Supervised Random Forest | 0.9528 | 0.8938 | 0.9224 | 0.9029 | 0.9688 | 0.9883 | 12.02 |
| Hybrid | 0.9570 | 0.7876 | 0.8641 | 0.8377 | 0.9383 | 0.9713 | 9.62 |

### Official hybrid event-level results

| Metric | Value |
|---|---:|
| True anomaly events | 107 |
| Detected events | 88 |
| Missed events | 19 |
| False-alert events | 4 |
| Event precision | 0.9565 |
| Event recall | 0.8224 |
| Event F1 | 0.8844 |

Event groups are formed from temporally adjacent positive segments on the same channel. Event metrics complement, rather than replace, segment-level metrics.

---

## Processed Data Layout

```text
data/opssat/
├── raw/
│   ├── dataset.csv
│   └── segments.csv
├── processed/
│   ├── train_features.csv
│   ├── validation_features.csv
│   ├── test_features.csv
│   └── official_test_predictions.csv
└── upload_samples/
    ├── opssat_real_normal.csv
    ├── opssat_real_anomaly.csv
    ├── opssat_real_mixed.csv
    ├── opssat_real_magnetometer_anomalies.csv
    └── opssat_real_photodiode_anomalies.csv
```

The internal training and validation files come only from the official training partition. `test_features.csv` remains the untouched official test partition.

---

## Application Workspaces

- **Mission Overview** — overall risk, anomaly distribution, channel analytics, and data compatibility.
- **Telemetry Explorer** — original real signal values for a selected segment.
- **Incident Intelligence** — local feature-deviation evidence, model scores, and decision margin.
- **Upload & Test** — schema validation, duplicate removal, label-aware evaluation, confusion matrix, and event ledger.
- **Model Validation** — official test metrics, confusion matrix, split distribution, and event evaluation.
- **Data Drift Monitor** — distribution-shift comparison against nominal official-training data.
- **Reports & Responsible AI** — TXT, HTML, and analyzed CSV exports with limitations.
- **Dataset & Attribution** — data card, channel mapping, schema, license, and citation.
- **IBM Bob Evidence** — development log for challenge submission evidence.

The interface includes high-contrast **Dark** and **Light** modes.

---

## Upload Validation

Before inference, the application checks:

- required columns;
- numeric segment identifiers;
- valid timestamps and telemetry values;
- exact duplicate rows;
- empty channel identifiers;
- one channel per segment;
- minimum sample warnings;
- binary label validity;
- known versus unseen telemetry channels;
- label coverage;
- distribution drift relative to training data.

Unknown channels are accepted because the encoder can ignore unseen categories, but the interface clearly marks reduced reliability.

---

## Ground-Truth and Event Evaluation

When an uploaded CSV contains `anomaly` labels, the website calculates:

- Accuracy and balanced accuracy
- Precision, recall, and F1
- MCC, PR-AUC, and ROC-AUC
- Confusion matrix
- False alarms and missed anomalies
- Event precision, recall, and F1
- Event detection ledger with detected, missed, and false-alert events

When labels are absent, the application reports predictions without claiming accuracy.

---

## Telemetry Drift Monitor

The drift monitor compares uploaded engineered features with the nominal training envelope using:

- standardized mean shift;
- spread ratio;
- per-feature drift score;
- stable, moderate, and high drift bands;
- unknown-channel detection;
- overall compatibility status.

Distribution drift is **not** treated as proof of a spacecraft fault. It is a warning that model reliability may have changed.

---

## Upload Schemas

### Raw telemetry segments

Required:

```text
channel,timestamp,value
```

Recommended:

```text
channel,timestamp,value,label,sampling,anomaly,segment,train
```

Each segment must contain exactly one telemetry channel. If `segment` is omitted, the whole file is treated as one segment.

### Engineered segment features

The official `dataset.csv` schema is also accepted:

```text
segment,anomaly,train,channel,sampling,duration,len,mean,var,std,
kurtosis,skew,n_peaks,smooth10_n_peaks,smooth20_n_peaks,
diff_peaks,diff2_peaks,diff_var,diff2_var,gaps_squared,
len_weighted,var_div_duration,var_div_len
```

`anomaly` and `train` may be omitted for unlabeled external predictions.

---

## Quick Start

### Docker + PostgreSQL + pgAdmin

For local Windows deployment, create the environment file from the provided template:

```powershell
copy .env.local.example .env
```

For a Linux server deployment:

```bash
cp .env.server.example .env
```

Open `.env` and replace every `CHANGE_ME` value with secure values before starting the stack.

Then run:

```bash
docker compose up -d --build
```

The stack starts the Streamlit app, PostgreSQL, and pgAdmin.

The application creates the `missionguard` schema, verifies all required tables, and seeds the official OPS-SAT reference dataset and model registry without duplicating them on later restarts.

See `DEPLOYMENT_PGADMIN_AR.md` for the complete Arabic deployment guide.

Default local endpoints:

- MissionGuard: `http://localhost:8501`
- pgAdmin: `http://localhost:5050`

### Windows

Full Docker stack with PostgreSQL and pgAdmin:

```text
START_DOCKER_WINDOWS.bat
```

`RUN_WINDOWS.bat` is only for local Streamlit mode and does not start pgAdmin.

### Manual setup

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

The packaged models are loaded automatically. Retraining is optional.

### Retrain and regenerate all artifacts

```bash
python train_models.py
```

Equivalent advanced command:

```bash
python scripts/train_opssat.py
```

### Run tests

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Production Deployment

The current public deployment uses:

```text
AWS Lightsail
Docker Compose
PostgreSQL
pgAdmin
Nginx
Let's Encrypt
DuckDNS
```

Public URL:

```text
https://missionguardplatform.duckdns.org
```

The production services are bound as follows:

```text
MissionGuard: 127.0.0.1:8501
PostgreSQL:   127.0.0.1:55432
pgAdmin:      127.0.0.1:5050
```

Only Nginx is publicly accessible through:

```text
TCP 80
TCP 443
```

PostgreSQL, pgAdmin, and Streamlit are not exposed directly to the internet.

### Production commands

Check container status:

```bash
cd /home/ubuntu/missionguard-production
docker compose ps
```

View recent logs:

```bash
docker compose logs --tail=100
```

Restart the stack:

```bash
docker compose restart
```

Rebuild after uploading code changes:

```bash
docker compose up -d --build
```

Check Nginx:

```bash
sudo systemctl is-active nginx
sudo systemctl is-enabled nginx
```

Test HTTPS certificate renewal:

```bash
sudo certbot renew --dry-run
```

---

## Updating the Deployment

GitHub and the AWS deployment are currently separate.

Pushing code to GitHub does not automatically update the running AWS server.

To publish a new version:

1. Commit and push the local changes to GitHub.

```bash
git add .
git commit -m "Describe the update"
git push
```

2. Upload the modified files to:

```text
/home/ubuntu/missionguard-production
```

3. Rebuild the application on the server:

```bash
cd /home/ubuntu/missionguard-production
docker compose up -d --build
```

4. Verify the services:

```bash
docker compose ps
```

---

## Security Notes

Never commit or publicly share:

```text
.env
*.pem
*.ppk
*.key
missionguard-credentials.txt
.streamlit/secrets.toml
```

Recommended public firewall ports:

```text
22   SSH
80   HTTP
443  HTTPS
```

Do not expose these ports publicly:

```text
8501
5050
5432
55432
```

The `.env` file must remain local to the development machine or production server.

Only example files such as the following should be committed:

```text
.env.example
.env.local.example
.env.server.example
```

---

## Responsible-AI Limitations

- The model predicts **segment-level statistical anomalies**, not confirmed hardware root causes.
- It uses nine selected OPS-SAT channels, not every spacecraft subsystem.
- A high benchmark score does not make the system certified flight software.
- High-risk or low-margin outputs require human review.
- Drift estimates can be unstable for very small uploads.
- Binary labels do not identify anomaly subtype or engineering cause.

---

## Attribution

This project uses OPSSAT-AD under the Creative Commons Attribution 4.0 International license:

Bogdan Ruszczak, Krzysztof Kotowski, Jakub Nalepa, and David Evans, **OPSSAT-AD — anomaly detection dataset for satellite telemetry**, Zenodo, DOI: `10.5281/zenodo.15108715`.

---

## PostgreSQL Connection Modes

MissionGuard accepts either a managed PostgreSQL connection string or separate connection settings. `DATABASE_URL` takes priority when both are present.

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require
POSTGRES_SCHEMA=missionguard
```

For local PostgreSQL or Docker Compose, use:

```env
POSTGRES_USER=missionguard
POSTGRES_PASSWORD=CHANGE_ME
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=missionguard_ai
POSTGRES_SCHEMA=missionguard
```

Server environment variables take priority over values stored in a local `.env` file.

Useful database commands:

```bash
python scripts/bootstrap_database.py
python scripts/initialize_database.py
python scripts/test_database.py
```

---

## Deployment Status

MissionGuard AI is currently:

- deployed on AWS Lightsail;
- running through Docker Compose;
- connected to PostgreSQL;
- protected behind Nginx;
- available through HTTPS;
- configured to restart after server reboots;
- backed up using an AWS Lightsail snapshot;
- available publicly through the MissionGuard DuckDNS address.
