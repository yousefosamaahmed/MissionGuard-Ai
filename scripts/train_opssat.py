from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.opssat import (  # noqa: E402
    FEATURE_SCHEMA,
    evaluate_event_detection,
    normalize_feature_frame,
    save_artifact,
    train_opssat_artifact,
)

DATASET = PROJECT_ROOT / "data" / "opssat" / "raw" / "dataset.csv"
SEGMENTS = PROJECT_ROOT / "data" / "opssat" / "raw" / "segments.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "opssat_model.joblib"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "opssat" / "processed" / "official_test_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "models" / "opssat_metrics.csv"
METADATA_PATH = PROJECT_ROOT / "models" / "opssat_metadata.json"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "opssat_feature_columns.json"
ISOLATION_BUNDLE_PATH = PROJECT_ROOT / "models" / "opssat_isolation_bundle.joblib"
SUPERVISED_BUNDLE_PATH = PROJECT_ROOT / "models" / "opssat_supervised_bundle.joblib"
PROCESSED_DIR = PROJECT_ROOT / "data" / "opssat" / "processed"


def _write_processed_splits(data: pd.DataFrame, artifact: dict) -> None:
    features, _ = normalize_feature_frame(data)
    fit_ids = set(map(int, artifact["internal_fit_segment_ids"]))
    validation_ids = set(map(int, artifact["validation_segment_ids"]))
    train_features = features[(features["train"] == 1) & features["segment"].isin(fit_ids)].copy()
    validation_features = features[(features["train"] == 1) & features["segment"].isin(validation_ids)].copy()
    test_features = features[features["train"] == 0].copy()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_features.to_csv(PROCESSED_DIR / "train_features.csv", index=False)
    validation_features.to_csv(PROCESSED_DIR / "validation_features.csv", index=False)
    test_features.to_csv(PROCESSED_DIR / "test_features.csv", index=False)


def _write_component_bundles(artifact: dict) -> None:
    isolation_bundle = {
        "artifact_version": artifact["artifact_version"],
        "source": artifact["source"],
        "preprocessor": artifact["isolation_preprocessor"],
        "model": artifact["isolation_model"],
        "calibration": artifact["isolation_calibration"],
        "threshold": artifact["thresholds"]["isolation"],
        "numeric_features": artifact["numeric_features"],
        "channels": artifact["channels"],
    }
    supervised_bundle = {
        "artifact_version": artifact["artifact_version"],
        "source": artifact["source"],
        "preprocessor": artifact["supervised_preprocessor"],
        "model": artifact["supervised_model"],
        "threshold": artifact["thresholds"]["supervised"],
        "numeric_features": artifact["numeric_features"],
        "channels": artifact["channels"],
    }
    joblib.dump(isolation_bundle, ISOLATION_BUNDLE_PATH)
    joblib.dump(supervised_bundle, SUPERVISED_BUNDLE_PATH)


def main() -> None:
    if not DATASET.exists():
        raise SystemExit(f"Missing real OPSSAT dataset: {DATASET}")
    data = pd.read_csv(DATASET)
    artifact, predictions, metrics = train_opssat_artifact(data)
    save_artifact(artifact, MODEL_PATH)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    _write_processed_splits(data, artifact)
    _write_component_bundles(artifact)

    feature_payload = {
        "feature_schema": FEATURE_SCHEMA,
        "model_numeric_features": artifact["numeric_features"],
        "excluded_from_model": ["segment", "anomaly", "train", "channel"],
        "channel_feature": "channel",
    }
    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_payload, indent=2), encoding="utf-8")

    metadata_keys = {
        "artifact_version",
        "source",
        "doi",
        "license",
        "thresholds",
        "hybrid_weights",
        "numeric_features",
        "feature_schema",
        "official_train_rows",
        "official_test_rows",
        "internal_fit_rows",
        "internal_validation_rows",
        "channels",
        "training_channel_distribution",
        "random_state",
        "threshold_selection",
    }
    metadata = {key: value for key, value in artifact.items() if key in metadata_keys}
    metadata["test_metrics"] = metrics.to_dict(orient="records")

    if SEGMENTS.exists():
        segments = pd.read_csv(SEGMENTS)
        test_segments = segments[segments["train"] == 0].copy()
        event_metrics, _ = evaluate_event_detection(predictions, test_segments)
        metadata["official_test_event_metrics"] = event_metrics

    metadata["packaged_artifacts"] = {
        "combined_model": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "isolation_bundle": str(ISOLATION_BUNDLE_PATH.relative_to(PROJECT_ROOT)),
        "supervised_bundle": str(SUPERVISED_BUNDLE_PATH.relative_to(PROJECT_ROOT)),
        "feature_columns": str(FEATURE_COLUMNS_PATH.relative_to(PROJECT_ROOT)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(metrics.to_string(index=False))
    print(f"Saved combined model: {MODEL_PATH}")
    print(f"Saved component bundles: {ISOLATION_BUNDLE_PATH.name}, {SUPERVISED_BUNDLE_PATH.name}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
    print(f"Saved processed train/validation/test files in: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
