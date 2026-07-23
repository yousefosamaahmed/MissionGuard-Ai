from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_SCHEMA = [
    "segment",
    "anomaly",
    "train",
    "channel",
    "sampling",
    "duration",
    "len",
    "mean",
    "var",
    "std",
    "kurtosis",
    "skew",
    "n_peaks",
    "smooth10_n_peaks",
    "smooth20_n_peaks",
    "diff_peaks",
    "diff2_peaks",
    "diff_var",
    "diff2_var",
    "gaps_squared",
    "len_weighted",
    "var_div_duration",
    "var_div_len",
]

RAW_REQUIRED = {"channel", "timestamp", "value"}
MODEL_EXCLUDED = {"segment", "anomaly", "train", "channel"}
MIN_RAW_SAMPLES = 8
MAX_INVALID_ROW_FRACTION = 0.50

CHANNEL_NAMES = {
    "CADC0872": "Magnetometer X",
    "CADC0873": "Magnetometer Y",
    "CADC0874": "Magnetometer Z",
    "CADC0884": "Photodiode 1",
    "CADC0886": "Photodiode 2",
    "CADC0888": "Photodiode 3",
    "CADC0890": "Photodiode 4",
    "CADC0892": "Photodiode 5",
    "CADC0894": "Photodiode 6",
}


@dataclass(frozen=True)
class UploadValidation:
    kind: str
    rows: int
    segments: int
    channels: int
    messages: tuple[str, ...]
    removed_rows: int = 0
    duplicate_rows: int = 0
    label_coverage: float = 0.0


def _number_of_peaks(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return 0
    value_range = float(np.max(values) - np.min(values))
    prominence = 0.1 * value_range
    return int(len(signal.find_peaks(values, prominence=prominence)[0]))


def _smooth_peaks(values: np.ndarray, window: int) -> int:
    if len(values) < 2:
        return 0
    safe_window = min(max(int(window), 1), len(values))
    kernel = np.ones(safe_window, dtype=float) / safe_window
    convolved = np.convolve(values, kernel, mode="same")
    return _number_of_peaks(convolved)


def _safe_variance(values: np.ndarray, order: int = 0) -> float:
    transformed = np.diff(values, n=order) if order else values
    transformed = transformed[np.isfinite(transformed)]
    return float(np.var(transformed)) if transformed.size else 0.0


def _safe_peaks(values: np.ndarray, order: int = 0) -> int:
    transformed = np.diff(values, n=order) if order else values
    return _number_of_peaks(transformed)


def _safe_shape_statistic(function: Any, values: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        result = float(function(values)) if len(values) else 0.0
    return result if np.isfinite(result) else 0.0


def _coerce_binary_label(series: pd.Series, column_name: str) -> tuple[pd.Series, int]:
    mapped = series.copy()
    if mapped.dtype == object or pd.api.types.is_string_dtype(mapped):
        mapped = mapped.astype(str).str.strip().str.lower().replace(
            {
                "normal": 0,
                "nominal": 0,
                "false": 0,
                "no": 0,
                "anomaly": 1,
                "anomalous": 1,
                "true": 1,
                "yes": 1,
                "nan": np.nan,
                "none": np.nan,
                "": np.nan,
            }
        )
    numeric = pd.to_numeric(mapped, errors="coerce")
    invalid_mask = numeric.notna() & ~numeric.isin([0, 1])
    invalid_count = int(invalid_mask.sum())
    numeric.loc[invalid_mask] = np.nan
    return numeric, invalid_count


def normalize_raw_segments(frame: pd.DataFrame) -> tuple[pd.DataFrame, UploadValidation]:
    data = frame.copy()
    data.columns = data.columns.astype(str).str.strip()
    missing = RAW_REQUIRED - set(data.columns)
    if missing:
        raise ValueError("Missing raw OPSSAT columns: " + ", ".join(sorted(missing)))

    original_rows = len(data)
    messages: list[str] = []
    if "segment" not in data.columns:
        data["segment"] = 1
        messages.append("No segment column was supplied; all rows were assigned to segment 1.")
    if "sampling" not in data.columns:
        data["sampling"] = np.nan
    if "anomaly" not in data.columns:
        data["anomaly"] = np.nan
    if "train" not in data.columns:
        data["train"] = np.nan
    if "label" not in data.columns:
        data["label"] = "unknown"

    duplicate_subset = [c for c in ["segment", "channel", "timestamp", "value"] if c in data.columns]
    duplicate_rows = int(data.duplicated(subset=duplicate_subset).sum())
    if duplicate_rows:
        data = data.drop_duplicates(subset=duplicate_subset).copy()
        messages.append(f"Removed {duplicate_rows} duplicate telemetry row(s).")

    data["channel"] = data["channel"].astype(str).str.strip()
    blank_channel = data["channel"].isin(["", "nan", "None"])
    if blank_channel.any():
        count = int(blank_channel.sum())
        data = data.loc[~blank_channel].copy()
        messages.append(f"Removed {count} row(s) with an empty channel identifier.")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data["value"] = data["value"].replace([np.inf, -np.inf], np.nan)
    invalid_mask = data[["timestamp", "value"]].isna().any(axis=1)
    invalid = int(invalid_mask.sum())
    if original_rows and invalid / original_rows > MAX_INVALID_ROW_FRACTION:
        raise ValueError(
            f"More than {MAX_INVALID_ROW_FRACTION:.0%} of telemetry rows have invalid timestamps or values."
        )
    if invalid:
        data = data.loc[~invalid_mask].copy()
        messages.append(f"Removed {invalid} invalid timestamp/value row(s).")
    if data.empty:
        raise ValueError("The uploaded raw telemetry file has no valid rows.")

    data["segment"] = pd.to_numeric(data["segment"], errors="coerce")
    if data["segment"].isna().any():
        raise ValueError("Segment identifiers must be numeric.")
    data["segment"] = data["segment"].astype(int)
    data["sampling"] = pd.to_numeric(data["sampling"], errors="coerce")
    data["anomaly"], invalid_labels = _coerce_binary_label(data["anomaly"], "anomaly")
    data["train"], invalid_split = _coerce_binary_label(data["train"], "train")
    if invalid_labels:
        messages.append(f"Ignored {invalid_labels} invalid ground-truth label value(s); only 0/1 are accepted.")
    if invalid_split:
        messages.append(f"Ignored {invalid_split} invalid train/test split value(s); only 0/1 are accepted.")

    data = data.sort_values(["segment", "timestamp"]).reset_index(drop=True)

    for segment_id, group in data.groupby("segment", sort=False):
        if group["channel"].nunique() != 1:
            raise ValueError(
                f"Segment {segment_id} contains multiple channels. "
                "OPSSAT-AD defines one telemetry channel per segment."
            )
        if len(group) < MIN_RAW_SAMPLES:
            messages.append(
                f"Segment {segment_id} contains fewer than {MIN_RAW_SAMPLES} samples; "
                "its engineered statistics may be unstable."
            )
        duplicate_timestamps = int(group["timestamp"].duplicated().sum())
        if duplicate_timestamps:
            messages.append(
                f"Segment {segment_id} contains {duplicate_timestamps} repeated timestamp(s)."
            )
        if group["sampling"].isna().all() or (group["sampling"].dropna() <= 0).all():
            diffs = group["timestamp"].diff().dropna().dt.total_seconds()
            inferred = int(round(float(diffs.median()))) if not diffs.empty else 1
            data.loc[group.index, "sampling"] = max(inferred, 1)
        else:
            valid_sampling = group["sampling"].dropna()
            replacement = float(valid_sampling[valid_sampling > 0].median()) if (valid_sampling > 0).any() else 1.0
            data.loc[group.index, "sampling"] = group["sampling"].fillna(replacement).clip(lower=1)

    label_coverage = float(data["anomaly"].notna().mean())
    validation = UploadValidation(
        kind="raw_segments",
        rows=len(data),
        segments=int(data["segment"].nunique()),
        channels=int(data["channel"].nunique()),
        messages=tuple(dict.fromkeys(messages)),
        removed_rows=max(original_rows - len(data), 0),
        duplicate_rows=duplicate_rows,
        label_coverage=label_coverage,
    )
    return data, validation


def _features_from_normalized_segments(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for segment_id, group in data.groupby("segment", sort=True):
        group = group.sort_values("timestamp")
        values = group["value"].to_numpy(dtype=float)
        start = group["timestamp"].iloc[0]
        end = group["timestamp"].iloc[-1]
        duration_seconds = int(max((end - start).total_seconds(), 0))
        diffs = group["timestamp"].diff().dropna().dt.total_seconds()
        gaps_squared = float(np.square(diffs.to_numpy(dtype=float)).sum())
        sampling_values = pd.to_numeric(group["sampling"], errors="coerce").dropna()
        sampling = int(round(float(sampling_values.iloc[0]))) if not sampling_values.empty else 1
        anomaly_values = pd.to_numeric(group["anomaly"], errors="coerce").dropna()
        train_values = pd.to_numeric(group["train"], errors="coerce").dropna()

        row = {
            "segment": int(segment_id),
            "anomaly": int(anomaly_values.iloc[0]) if not anomaly_values.empty else np.nan,
            "train": int(train_values.iloc[0]) if not train_values.empty else np.nan,
            "channel": str(group["channel"].iloc[0]),
            "sampling": max(sampling, 1),
            "duration": duration_seconds,
            "len": int(len(values)),
            "mean": float(np.mean(values)),
            "var": float(np.var(values)),
            "std": float(np.std(values)),
            "kurtosis": _safe_shape_statistic(kurtosis, values),
            "skew": _safe_shape_statistic(skew, values),
            "n_peaks": _number_of_peaks(values),
            "smooth10_n_peaks": _smooth_peaks(values, 10),
            "smooth20_n_peaks": _smooth_peaks(values, 20),
            "diff_peaks": _safe_peaks(values, 1),
            "diff2_peaks": _safe_peaks(values, 2),
            "diff_var": _safe_variance(values, 1),
            "diff2_var": _safe_variance(values, 2),
            "gaps_squared": gaps_squared,
        }
        row["len_weighted"] = row["sampling"] * row["len"]
        row["var_div_duration"] = row["var"] / max(row["duration"], 1)
        row["var_div_len"] = row["var"] / max(row["len"], 1)
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_SCHEMA)


def features_from_segments(frame: pd.DataFrame) -> pd.DataFrame:
    data, _ = normalize_raw_segments(frame)
    return _features_from_normalized_segments(data)


def normalize_feature_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, UploadValidation]:
    data = frame.copy()
    data.columns = data.columns.astype(str).str.strip()
    required = set(FEATURE_SCHEMA) - {"anomaly", "train"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError("Missing OPSSAT feature columns: " + ", ".join(sorted(missing)))

    original_rows = len(data)
    messages: list[str] = []
    if "anomaly" not in data.columns:
        data["anomaly"] = np.nan
    if "train" not in data.columns:
        data["train"] = np.nan
    data = data.reindex(columns=FEATURE_SCHEMA)

    duplicate_rows = int(data.duplicated(subset=["segment", "channel"]).sum())
    if duplicate_rows:
        data = data.drop_duplicates(subset=["segment", "channel"], keep="first").copy()
        messages.append(f"Removed {duplicate_rows} duplicate segment-feature row(s).")

    data["channel"] = data["channel"].astype(str).str.strip()
    blank_channel = data["channel"].isin(["", "nan", "None"])
    if blank_channel.any():
        count = int(blank_channel.sum())
        data = data.loc[~blank_channel].copy()
        messages.append(f"Removed {count} row(s) with an empty channel identifier.")

    for column in [c for c in FEATURE_SCHEMA if c not in {"channel", "anomaly", "train"}]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["anomaly"], invalid_labels = _coerce_binary_label(data["anomaly"], "anomaly")
    data["train"], invalid_split = _coerce_binary_label(data["train"], "train")
    if invalid_labels:
        messages.append(f"Ignored {invalid_labels} invalid ground-truth label value(s); only 0/1 are accepted.")
    if invalid_split:
        messages.append(f"Ignored {invalid_split} invalid train/test split value(s); only 0/1 are accepted.")

    numeric_columns = [c for c in FEATURE_SCHEMA if c != "channel"]
    data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)
    required_numeric = [c for c in FEATURE_SCHEMA if c not in {"anomaly", "train", "channel"}]
    invalid_mask = data[required_numeric].isna().any(axis=1)
    invalid = int(invalid_mask.sum())
    if original_rows and invalid / original_rows > MAX_INVALID_ROW_FRACTION:
        raise ValueError(
            f"More than {MAX_INVALID_ROW_FRACTION:.0%} of engineered rows contain invalid required values."
        )
    if invalid:
        data = data.loc[~invalid_mask].copy()
        messages.append(f"Removed {invalid} invalid feature row(s).")
    if data.empty:
        raise ValueError("The uploaded feature file has no valid OPSSAT rows.")

    data["segment"] = data["segment"].astype(int)
    label_coverage = float(data["anomaly"].notna().mean())
    validation = UploadValidation(
        kind="segment_features",
        rows=len(data),
        segments=int(data["segment"].nunique()),
        channels=int(data["channel"].nunique()),
        messages=tuple(dict.fromkeys(messages)),
        removed_rows=max(original_rows - len(data), 0),
        duplicate_rows=duplicate_rows,
        label_coverage=label_coverage,
    )
    return data, validation


def detect_and_prepare_upload(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame | None, UploadValidation]:
    columns = set(frame.columns.astype(str).str.strip())
    if RAW_REQUIRED.issubset(columns):
        raw, validation = normalize_raw_segments(frame)
        return _features_from_normalized_segments(raw), raw, validation
    if {"channel", "mean", "var", "std", "segment"}.issubset(columns):
        features, validation = normalize_feature_frame(frame)
        return features, None, validation
    raise ValueError(
        "Unsupported CSV schema. Upload either OPSSAT segments.csv-style raw telemetry "
        "or dataset.csv-style engineered segment features."
    )


def validate_features_against_artifact(frame: pd.DataFrame, artifact: dict[str, Any]) -> dict[str, Any]:
    features, _ = normalize_feature_frame(frame)
    trained_channels = set(map(str, artifact.get("channels", [])))
    incoming_channels = set(features["channel"].astype(str))
    unknown_channels = sorted(incoming_channels - trained_channels)
    known_ratio = float(features["channel"].isin(trained_channels).mean()) if trained_channels else 0.0
    label_coverage = float(features["anomaly"].notna().mean())
    warnings: list[str] = []
    if len(features) < 3:
        warnings.append("Fewer than three segments were supplied; aggregate validation and drift estimates are unstable.")
    if unknown_channels:
        warnings.append(
            "Unknown telemetry channel(s) are not represented in model training: "
            + ", ".join(unknown_channels)
            + ". Predictions are allowed, but reliability is reduced."
        )
    if label_coverage == 0:
        warnings.append("No ground-truth anomaly labels are available; predictive accuracy cannot be scored.")
    elif label_coverage < 1:
        warnings.append(f"Ground-truth labels are available for only {label_coverage:.1%} of segments.")

    if known_ratio == 1 and len(features) >= 3:
        status = "Good"
    elif known_ratio >= 0.75:
        status = "Review"
    else:
        status = "Poor"
    return {
        "status": status,
        "rows": len(features),
        "segments": int(features["segment"].nunique()),
        "known_channel_ratio": known_ratio,
        "unknown_channels": unknown_channels,
        "label_coverage": label_coverage,
        "warnings": warnings,
    }


def _build_preprocessor(frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    numeric = [column for column in frame.columns if column not in MODEL_EXCLUDED]
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("channel", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["channel"]),
        ],
        remainder="drop",
    )
    return preprocessor, numeric


def _calibrate(raw: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    low = float(np.quantile(reference, 0.50))
    high = float(np.quantile(reference, 0.995))
    if high <= low:
        high = low + max(float(np.std(reference)), 1e-9)
    scores = np.clip((raw - low) / (high - low) * 100, 0, 100)
    return scores, (low, high)


def _apply_calibration(raw: np.ndarray, calibration: tuple[float, float]) -> np.ndarray:
    low, high = calibration
    return np.clip((raw - low) / max(high - low, 1e-9) * 100, 0, 100)


def _best_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    candidates = np.linspace(5, 95, 181)
    f1_values = [f1_score(y_true, score >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(f1_values))])


def train_opssat_artifact(
    data: pd.DataFrame, random_state: int = 42
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    features, _ = normalize_feature_frame(data)
    official_train = features[features["train"] == 1].copy()
    official_test = features[features["train"] == 0].copy()
    if official_train.empty or official_test.empty:
        raise ValueError("The official OPSSAT train/test split is missing.")

    train_part, validation_part = train_test_split(
        official_train,
        test_size=0.25,
        random_state=random_state,
        stratify=official_train["anomaly"],
    )

    # Thresholds are selected on internal validation only; the official test remains unseen.
    iso_pre_val, numeric_features = _build_preprocessor(train_part)
    nominal_train_part = train_part[train_part["anomaly"] == 0]
    x_nominal_val = iso_pre_val.fit_transform(nominal_train_part)
    x_validation_iso = iso_pre_val.transform(validation_part)
    iso_val = IsolationForest(
        n_estimators=350,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    ).fit(x_nominal_val)
    raw_nominal_val = -iso_val.decision_function(x_nominal_val)
    iso_validation, _ = _calibrate(-iso_val.decision_function(x_validation_iso), raw_nominal_val)

    sup_pre_val, _ = _build_preprocessor(train_part)
    x_train_sup_val = sup_pre_val.fit_transform(train_part)
    x_validation_sup = sup_pre_val.transform(validation_part)
    sup_val = RandomForestClassifier(
        n_estimators=450,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    ).fit(x_train_sup_val, train_part["anomaly"].astype(int))
    sup_validation = sup_val.predict_proba(x_validation_sup)[:, 1] * 100
    hybrid_validation = 0.42 * iso_validation + 0.58 * sup_validation

    thresholds = {
        "isolation": _best_threshold(validation_part["anomaly"].to_numpy(), iso_validation),
        "supervised": _best_threshold(validation_part["anomaly"].to_numpy(), sup_validation),
        "hybrid": _best_threshold(validation_part["anomaly"].to_numpy(), hybrid_validation),
    }

    # Final models fit on the complete official training split after threshold selection.
    iso_pre, _ = _build_preprocessor(official_train)
    nominal_full = official_train[official_train["anomaly"] == 0]
    x_nominal = iso_pre.fit_transform(nominal_full)
    isolation = IsolationForest(
        n_estimators=500,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    ).fit(x_nominal)
    raw_nominal = -isolation.decision_function(x_nominal)
    _, isolation_calibration = _calibrate(raw_nominal, raw_nominal)

    sup_pre, _ = _build_preprocessor(official_train)
    x_train_sup = sup_pre.fit_transform(official_train)
    classifier = RandomForestClassifier(
        n_estimators=600,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    ).fit(x_train_sup, official_train["anomaly"].astype(int))

    nominal_numeric = nominal_full[numeric_features]
    channel_distribution = official_train["channel"].value_counts(normalize=True).to_dict()
    artifact: dict[str, Any] = {
        "artifact_version": "4.0.0-opssat",
        "source": "OPSSAT-AD v2",
        "doi": "10.5281/zenodo.15108715",
        "license": "CC BY 4.0",
        "isolation_preprocessor": iso_pre,
        "isolation_model": isolation,
        "isolation_calibration": isolation_calibration,
        "supervised_preprocessor": sup_pre,
        "supervised_model": classifier,
        "thresholds": thresholds,
        "hybrid_weights": {"isolation": 0.42, "supervised": 0.58},
        "numeric_features": numeric_features,
        "feature_schema": FEATURE_SCHEMA,
        "normal_feature_mean": nominal_numeric.mean().to_dict(),
        "normal_feature_std": nominal_numeric.std().replace(0, 1.0).fillna(1.0).to_dict(),
        "normal_feature_median": nominal_numeric.median().to_dict(),
        "normal_feature_q25": nominal_numeric.quantile(0.25).to_dict(),
        "normal_feature_q75": nominal_numeric.quantile(0.75).to_dict(),
        "official_train_rows": len(official_train),
        "official_test_rows": len(official_test),
        "internal_fit_rows": len(train_part),
        "internal_validation_rows": len(validation_part),
        "internal_fit_segment_ids": train_part["segment"].astype(int).tolist(),
        "validation_segment_ids": validation_part["segment"].astype(int).tolist(),
        "channels": sorted(features["channel"].unique().tolist()),
        "training_channel_distribution": channel_distribution,
        "random_state": random_state,
        "threshold_selection": "Stratified internal validation subset from official training split",
    }

    official_predictions = predict_feature_rows(official_test, artifact)
    metrics = evaluate_models(official_predictions, thresholds)
    return artifact, official_predictions, metrics


def predict_feature_rows(frame: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    features, _ = normalize_feature_frame(frame)
    result = features.copy()

    x_iso = artifact["isolation_preprocessor"].transform(features)
    raw_iso = -artifact["isolation_model"].decision_function(x_iso)
    isolation_score = _apply_calibration(raw_iso, artifact["isolation_calibration"])

    x_sup = artifact["supervised_preprocessor"].transform(features)
    supervised_score = artifact["supervised_model"].predict_proba(x_sup)[:, 1] * 100
    weights = artifact.get("hybrid_weights", {"isolation": 0.42, "supervised": 0.58})
    hybrid_score = weights["isolation"] * isolation_score + weights["supervised"] * supervised_score
    threshold = float(artifact["thresholds"]["hybrid"])

    result["isolation_score"] = np.round(isolation_score, 2)
    result["supervised_score"] = np.round(supervised_score, 2)
    result["hybrid_score"] = np.round(hybrid_score, 2)
    result["prediction"] = (hybrid_score >= threshold).astype(int)
    result["prediction_label"] = np.where(result["prediction"] == 1, "Anomaly", "Normal")

    watch_threshold = max(10.0, threshold - 18.0)
    critical_threshold = max(82.0, threshold + 18.0)
    result["risk_level"] = np.select(
        [hybrid_score >= critical_threshold, hybrid_score >= threshold, hybrid_score >= watch_threshold],
        ["Critical", "Warning", "Watch"],
        default="Normal",
    )
    maximum_possible_margin = max(threshold, 100.0 - threshold, 1.0)
    result["decision_margin"] = np.clip(
        np.abs(hybrid_score - threshold) / maximum_possible_margin * 100,
        0,
        100,
    ).round(1)
    # Kept only for backwards compatibility with older report exports.
    result["confidence"] = result["decision_margin"]

    means = pd.Series(artifact["normal_feature_mean"])
    stds = pd.Series(artifact["normal_feature_std"]).replace(0, 1.0)
    numeric = artifact["numeric_features"]
    deviations = ((features[numeric] - means[numeric]) / stds[numeric]).abs()
    contributions = deviations.div(deviations.sum(axis=1).replace(0, 1.0), axis=0) * 100
    result["top_feature"] = contributions.idxmax(axis=1)
    result["top_feature_contribution"] = contributions.max(axis=1).round(1)
    result["feature_contributions"] = contributions.round(2).to_dict(orient="records")
    result["explanation"] = result.apply(
        lambda row: (
            f"{row['top_feature']} is the strongest engineered deviation "
            f"({row['top_feature_contribution']:.1f}% of local deviation evidence); "
            f"the uncalibrated supervised model score is {row['supervised_score']:.1f}/100 and "
            f"the hybrid anomaly score is {row['hybrid_score']:.1f}/100."
        ),
        axis=1,
    )
    return result


def _safe_auc(y_true: np.ndarray, score: np.ndarray, kind: str) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if kind == "roc":
        return float(roc_auc_score(y_true, score))
    return float(average_precision_score(y_true, score))


def evaluate_binary_predictions(
    predictions: pd.DataFrame,
    prediction_column: str = "prediction",
    label_column: str = "anomaly",
    score_column: str = "hybrid_score",
) -> dict[str, Any] | None:
    labeled = predictions.dropna(subset=[label_column]).copy()
    if labeled.empty:
        return None
    y_true = labeled[label_column].astype(int).to_numpy()
    y_pred = labeled[prediction_column].astype(int).to_numpy()
    score = labeled[score_column].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Labeled Segments": int(len(labeled)),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "PR-AUC": _safe_auc(y_true, score, "pr"),
        "ROC-AUC": _safe_auc(y_true, score, "roc"),
        "False Alarms": int(fp),
        "Missed Anomalies": int(fn),
        "False Alarms / 1000": fp / max(tn + fp, 1) * 1000,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def evaluate_models(
    predictions: pd.DataFrame, thresholds: dict[str, float] | None = None
) -> pd.DataFrame:
    labeled = predictions.dropna(subset=["anomaly"]).copy()
    y_true = labeled["anomaly"].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for name, score_column, threshold_key in [
        ("OPSSAT Isolation Forest", "isolation_score", "isolation"),
        ("OPSSAT Supervised Random Forest", "supervised_score", "supervised"),
        ("OPSSAT Hybrid", "hybrid_score", "hybrid"),
    ]:
        score = labeled[score_column].to_numpy(dtype=float)
        threshold = float((thresholds or {}).get(threshold_key, 50.0))
        prediction = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
        rows.append(
            {
                "Model": name,
                "Precision": precision_score(y_true, prediction, zero_division=0),
                "Recall": recall_score(y_true, prediction, zero_division=0),
                "F1": f1_score(y_true, prediction, zero_division=0),
                "MCC": matthews_corrcoef(y_true, prediction),
                "PR-AUC": _safe_auc(y_true, score, "pr"),
                "ROC-AUC": _safe_auc(y_true, score, "roc"),
                "False Alarms / 1000": fp / max(tn + fp, 1) * 1000,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "Threshold": float(threshold),
            }
        )
    return pd.DataFrame(rows)


def _segment_boundaries(raw_segments: pd.DataFrame | None) -> pd.DataFrame | None:
    if raw_segments is None or raw_segments.empty:
        return None
    required = {"segment", "channel", "timestamp"}
    if not required.issubset(raw_segments.columns):
        return None
    raw = raw_segments.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    raw["sampling"] = pd.to_numeric(raw.get("sampling", 1), errors="coerce").fillna(1).clip(lower=1)
    raw = raw.dropna(subset=["timestamp"])
    if raw.empty:
        return None
    return (
        raw.groupby(["segment", "channel"], as_index=False)
        .agg(start_time=("timestamp", "min"), end_time=("timestamp", "max"), sampling=("sampling", "median"))
    )


def build_event_table(
    frame: pd.DataFrame,
    positive_column: str,
    raw_segments: pd.DataFrame | None = None,
    event_prefix: str = "E",
) -> pd.DataFrame:
    required = {"segment", "channel", positive_column}
    if not required.issubset(frame.columns):
        return pd.DataFrame(
            columns=["event_id", "channel", "first_segment", "last_segment", "segment_count", "segment_ids", "start_time", "end_time"]
        )
    rows = frame[["segment", "channel", positive_column]].drop_duplicates(["segment", "channel"]).copy()
    rows[positive_column] = pd.to_numeric(rows[positive_column], errors="coerce").fillna(0).astype(int)
    rows = rows[rows[positive_column] == 1].copy()
    if rows.empty:
        return pd.DataFrame(
            columns=["event_id", "channel", "first_segment", "last_segment", "segment_count", "segment_ids", "start_time", "end_time"]
        )

    boundaries = _segment_boundaries(raw_segments)
    if boundaries is not None:
        rows = rows.merge(boundaries, on=["segment", "channel"], how="left")
        rows = rows.sort_values(["channel", "start_time", "segment"], na_position="last")
    else:
        rows["start_time"] = pd.NaT
        rows["end_time"] = pd.NaT
        rows["sampling"] = 1.0
        rows = rows.sort_values(["channel", "segment"])

    event_rows: list[dict[str, Any]] = []
    event_counter = 0
    for channel, group in rows.groupby("channel", sort=True):
        current: list[pd.Series] = []
        previous: pd.Series | None = None
        for _, row in group.iterrows():
            same_event = False
            if previous is not None:
                if pd.notna(row["start_time"]) and pd.notna(previous["end_time"]):
                    gap_seconds = float((row["start_time"] - previous["end_time"]).total_seconds())
                    tolerance = max(60.0, 3.0 * max(float(row.get("sampling", 1)), float(previous.get("sampling", 1))))
                    same_event = gap_seconds <= tolerance
                else:
                    same_event = int(row["segment"]) == int(previous["segment"]) + 1
            if current and not same_event:
                event_counter += 1
                event_rows.append(_event_row(current, channel, f"{event_prefix}{event_counter:03d}"))
                current = []
            current.append(row)
            previous = row
        if current:
            event_counter += 1
            event_rows.append(_event_row(current, channel, f"{event_prefix}{event_counter:03d}"))

    return pd.DataFrame(event_rows)


def _event_row(rows: list[pd.Series], channel: str, event_id: str) -> dict[str, Any]:
    segments = tuple(sorted({int(row["segment"]) for row in rows}))
    starts = [row["start_time"] for row in rows if pd.notna(row["start_time"])]
    ends = [row["end_time"] for row in rows if pd.notna(row["end_time"])]
    return {
        "event_id": event_id,
        "channel": str(channel),
        "first_segment": min(segments),
        "last_segment": max(segments),
        "segment_count": len(segments),
        "segment_ids": segments,
        "start_time": min(starts) if starts else pd.NaT,
        "end_time": max(ends) if ends else pd.NaT,
    }


def evaluate_event_detection(
    predictions: pd.DataFrame,
    raw_segments: pd.DataFrame | None = None,
    label_column: str = "anomaly",
    prediction_column: str = "prediction",
) -> tuple[dict[str, Any] | None, pd.DataFrame]:
    if label_column not in predictions.columns or predictions[label_column].notna().sum() == 0:
        return None, pd.DataFrame()

    truth_events = build_event_table(predictions.dropna(subset=[label_column]), label_column, raw_segments, "GT")
    predicted_events = build_event_table(predictions, prediction_column, raw_segments, "P")

    truth_sets = [set(event) for event in truth_events.get("segment_ids", [])]
    pred_sets = [set(event) for event in predicted_events.get("segment_ids", [])]
    truth_matches: list[list[int]] = []
    for truth_index, truth_row in truth_events.iterrows():
        matches = [
            pred_index
            for pred_index, pred_row in predicted_events.iterrows()
            if str(pred_row["channel"]) == str(truth_row["channel"])
            and bool(truth_sets[truth_index] & pred_sets[pred_index])
        ]
        truth_matches.append(matches)

    matched_predicted = {index for matches in truth_matches for index in matches}
    detected_truth = sum(bool(matches) for matches in truth_matches)
    false_alert_events = len(predicted_events) - len(matched_predicted)
    event_precision = len(matched_predicted) / max(len(predicted_events), 1)
    event_recall = detected_truth / max(len(truth_events), 1)
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )

    ledger_rows: list[dict[str, Any]] = []
    for index, truth_row in truth_events.iterrows():
        matches = truth_matches[index]
        ledger_rows.append(
            {
                "Record Type": "Ground Truth Event",
                "Event ID": truth_row["event_id"],
                "Channel": truth_row["channel"],
                "Segments": ", ".join(map(str, truth_row["segment_ids"])),
                "Segment Count": truth_row["segment_count"],
                "Status": "Detected" if matches else "Missed",
                "Matched Prediction": ", ".join(predicted_events.loc[matches, "event_id"].tolist()) if matches else "—",
                "Start": truth_row["start_time"],
                "End": truth_row["end_time"],
            }
        )
    for index, pred_row in predicted_events.iterrows():
        if index not in matched_predicted:
            ledger_rows.append(
                {
                    "Record Type": "Predicted Event",
                    "Event ID": pred_row["event_id"],
                    "Channel": pred_row["channel"],
                    "Segments": ", ".join(map(str, pred_row["segment_ids"])),
                    "Segment Count": pred_row["segment_count"],
                    "Status": "False Alert",
                    "Matched Prediction": "—",
                    "Start": pred_row["start_time"],
                    "End": pred_row["end_time"],
                }
            )

    metrics = {
        "True Events": int(len(truth_events)),
        "Predicted Events": int(len(predicted_events)),
        "Detected Events": int(detected_truth),
        "Missed Events": int(len(truth_events) - detected_truth),
        "False Alert Events": int(false_alert_events),
        "Event Precision": float(event_precision),
        "Event Recall": float(event_recall),
        "Event F1": float(event_f1),
    }
    return metrics, pd.DataFrame(ledger_rows)


def assess_data_drift(
    frame: pd.DataFrame, artifact: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    features, _ = normalize_feature_frame(frame)
    numeric = list(artifact["numeric_features"])
    reference_mean = pd.Series(artifact["normal_feature_mean"], dtype=float).reindex(numeric)
    reference_std = (
        pd.Series(artifact["normal_feature_std"], dtype=float).reindex(numeric).replace(0, 1.0).fillna(1.0)
    )
    current_mean = features[numeric].mean()
    current_std = features[numeric].std(ddof=1).fillna(0.0)

    mean_shift = ((current_mean - reference_mean).abs() / reference_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(features) >= 5:
        spread_ratio = (current_std / reference_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        spread_shift = np.abs(np.log(spread_ratio.clip(lower=1e-6))).clip(upper=5.0)
    else:
        spread_ratio = pd.Series(1.0, index=numeric)
        spread_shift = pd.Series(0.0, index=numeric)
    drift_score = pd.concat([mean_shift.rename("mean"), spread_shift.rename("spread")], axis=1).max(axis=1)
    level = pd.cut(
        drift_score,
        bins=[-np.inf, 0.75, 1.5, np.inf],
        labels=["Stable", "Moderate", "High"],
        right=False,
    ).astype(str)

    details = pd.DataFrame(
        {
            "Feature": numeric,
            "Reference Mean": reference_mean.values,
            "Current Mean": current_mean.reindex(numeric).values,
            "Mean Shift (σ)": mean_shift.reindex(numeric).values,
            "Spread Ratio": spread_ratio.reindex(numeric).values,
            "Drift Score": drift_score.reindex(numeric).values,
            "Level": level.reindex(numeric).values,
        }
    ).sort_values("Drift Score", ascending=False)

    trained_channels = set(map(str, artifact.get("channels", [])))
    unknown_channels = sorted(set(features["channel"].astype(str)) - trained_channels)
    overall_score = float(details["Drift Score"].quantile(0.90)) if not details.empty else 0.0
    high_count = int((details["Level"] == "High").sum())
    moderate_count = int((details["Level"] == "Moderate").sum())
    if len(features) < 3:
        compatibility = "Insufficient Data"
    elif unknown_channels or high_count >= 3 or overall_score >= 2.0:
        compatibility = "Poor"
    elif high_count or moderate_count >= 3 or overall_score >= 0.75:
        compatibility = "Review"
    else:
        compatibility = "Good"

    notes: list[str] = []
    if len(features) < 5:
        notes.append("Drift estimates are based on fewer than five segments; spread comparisons were disabled.")
    if unknown_channels:
        notes.append("Unknown channel(s): " + ", ".join(unknown_channels))
    notes.append(
        "Drift compares uploaded engineered features with the nominal official-training envelope; "
        "distribution drift is not itself proof of a spacecraft fault."
    )
    summary = {
        "compatibility": compatibility,
        "overall_score": overall_score,
        "segments": len(features),
        "high_features": high_count,
        "moderate_features": moderate_count,
        "affected_features": details.loc[details["Level"].isin(["Moderate", "High"]), "Feature"].tolist(),
        "unknown_channels": unknown_channels,
        "notes": notes,
    }
    return summary, details


def save_artifact(artifact: dict[str, Any], path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, destination)
    return destination


def load_artifact(path: Path | str) -> dict[str, Any]:
    artifact = joblib.load(path)
    required = {"supervised_model", "isolation_model", "thresholds", "numeric_features"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise TypeError("Invalid OPSSAT model artifact.")
    return artifact
