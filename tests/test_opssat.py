from pathlib import Path

import pandas as pd
import pytest

from src.opssat import (
    assess_data_drift,
    detect_and_prepare_upload,
    evaluate_binary_predictions,
    evaluate_event_detection,
    load_artifact,
    predict_feature_rows,
    validate_features_against_artifact,
)

ROOT = Path(__file__).resolve().parents[1]


def test_packaged_real_files_exist():
    assert (ROOT / "data/opssat/raw/dataset.csv").exists()
    assert (ROOT / "data/opssat/raw/segments.csv").exists()
    assert (ROOT / "models/opssat_model.joblib").exists()
    assert (ROOT / "data/DATASET_CARD.md").exists()


def test_raw_upload_is_converted_and_predicted():
    raw = pd.read_csv(ROOT / "data/opssat/upload_samples/opssat_real_mixed.csv")
    features, normalized_raw, validation = detect_and_prepare_upload(raw)
    assert normalized_raw is not None
    assert validation.segments == 3
    assert len(features) == 3
    assert validation.label_coverage == 1.0
    artifact = load_artifact(ROOT / "models/opssat_model.joblib")
    predicted = predict_feature_rows(features, artifact)
    assert {
        "hybrid_score",
        "prediction",
        "risk_level",
        "decision_margin",
        "explanation",
    }.issubset(predicted.columns)


def test_feature_upload_is_supported():
    features = pd.read_csv(ROOT / "data/opssat/raw/dataset.csv").head(10)
    prepared, raw, validation = detect_and_prepare_upload(features)
    assert raw is None
    assert validation.kind == "segment_features"
    assert len(prepared) == 10


def test_upload_validation_rejects_missing_columns():
    with pytest.raises(ValueError, match="Unsupported CSV schema"):
        detect_and_prepare_upload(pd.DataFrame({"timestamp": ["2022-01-01"], "value": [1.0]}))


def test_duplicate_raw_rows_are_removed():
    raw = pd.read_csv(ROOT / "data/opssat/upload_samples/opssat_real_normal.csv")
    duplicated = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    _, normalized, validation = detect_and_prepare_upload(duplicated)
    assert normalized is not None
    assert validation.duplicate_rows == 1
    assert len(normalized) == len(raw)


def test_ground_truth_and_event_evaluation_are_available():
    raw = pd.read_csv(ROOT / "data/opssat/upload_samples/opssat_real_mixed.csv")
    features, normalized_raw, _ = detect_and_prepare_upload(raw)
    artifact = load_artifact(ROOT / "models/opssat_model.joblib")
    predicted = predict_feature_rows(features, artifact)
    row_metrics = evaluate_binary_predictions(predicted)
    event_metrics, ledger = evaluate_event_detection(predicted, normalized_raw)
    assert row_metrics is not None
    assert event_metrics is not None
    assert "F1" in row_metrics
    assert "Event F1" in event_metrics
    assert not ledger.empty


def test_drift_and_compatibility_monitor_return_results():
    features = pd.read_csv(ROOT / "data/opssat/raw/dataset.csv").head(20)
    artifact = load_artifact(ROOT / "models/opssat_model.joblib")
    compatibility = validate_features_against_artifact(features, artifact)
    drift_summary, drift_details = assess_data_drift(features, artifact)
    assert compatibility["status"] in {"Good", "Review", "Poor"}
    assert drift_summary["compatibility"] in {"Good", "Review", "Poor", "Insufficient Data"}
    assert {"Feature", "Drift Score", "Level"}.issubset(drift_details.columns)
