from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from database.repositories.analysis import (
    complete_analysis_run,
    create_analysis_run,
    create_incident,
    fail_analysis_run,
    save_prediction_batch,
)
from database.repositories.datasets import (
    create_dataset,
    create_dataset_file,
)
from database.repositories.telemetry import (
    list_session_feature_records,
    save_telemetry_batch,
)
from database.repositories.telemetry_sessions import (
    create_telemetry_session,
)
from src.mission_health import calculate_mission_health_score
from src.opssat import (
    load_artifact,
    predict_feature_rows,
)


@dataclass(frozen=True)
class OpsSatAnalysisResult:
    """
    Final result returned after completing
    a real OPS-SAT analysis.
    """

    analysis_run_id: UUID
    total_predictions: int
    total_anomalies: int
    total_incidents: int
    mean_risk_score: float
    maximum_risk_score: float
    mission_health_score: float
    mission_health_status: str


@dataclass(frozen=True)
class UploadedOpsSatPersistenceResult:
    """
    IDs and analysis metrics created for one uploaded CSV.
    """

    dataset_id: UUID
    original_dataset_file_id: UUID
    processed_dataset_file_id: UUID
    telemetry_session_id: UUID
    analysis_result: OpsSatAnalysisResult
    original_file_path: Path
    processed_file_path: Path
    sha256_hash: str


INCIDENT_SEVERITY_ORDER = {
    "Watch": 1,
    "Warning": 2,
    "Critical": 3,
}


def _require_dictionary(
    value: object,
    field_name: str,
) -> dict[str, Any]:
    """
    Validate and return a dictionary value.
    """

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(
            f"{field_name} must be a dictionary."
        )

    return value


def _safe_incident_token(
    value: str,
) -> str:
    """
    Convert a channel name into a safe incident-code token.
    """

    normalized = "".join(
        character
        if character.isalnum()
        else "-"
        for character in value.upper()
    )

    normalized = "-".join(
        part
        for part in normalized.split("-")
        if part
    )

    return normalized[:24] or "UNKNOWN"


def _build_inference_frame(
    records: list[dict[str, Any]],
    numeric_features: list[str],
) -> pd.DataFrame:
    """
    Reconstruct the OPS-SAT feature DataFrame
    from the stored PostgreSQL feature vectors.

    Ground-truth labels are not used for inference.
    """

    dataframe_rows: list[dict[str, Any]] = []

    for record in records:
        sample_metadata = _require_dictionary(
            record.get("sample_metadata"),
            "sample_metadata",
        )

        feature_values = _require_dictionary(
            record.get("feature_values"),
            "feature_values",
        )

        missing_features = [
            feature_name
            for feature_name in numeric_features
            if feature_name not in feature_values
        ]

        if missing_features:
            raise ValueError(
                "Stored feature vector is missing: "
                + ", ".join(missing_features)
            )

        segment = sample_metadata.get(
            "segment",
            record["sample_index"],
        )

        channel = str(
            sample_metadata.get(
                "channel",
                "unknown",
            )
        ).strip()

        train_flag = sample_metadata.get(
            "train_flag",
            0.0,
        )

        dataframe_row: dict[str, Any] = {
            "segment": segment,

            # The real label is deliberately excluded.
            # predict_feature_rows() preserves this column,
            # but it is not passed as a model feature.
            "anomaly": 0.0,

            "train": train_flag,
            "channel": channel,
        }

        for feature_name in numeric_features:
            dataframe_row[feature_name] = (
                feature_values[feature_name]
            )

        dataframe_rows.append(
            dataframe_row
        )

    dataframe = pd.DataFrame(
        dataframe_rows
    )

    expected_columns = [
        "segment",
        "anomaly",
        "train",
        "channel",
        *numeric_features,
    ]

    selected_frame = dataframe.loc[
        :,
        expected_columns,
    ].copy()

    selected_frame.reset_index(
        drop=True,
        inplace=True,
    )

    return selected_frame


def _build_prediction_records(
    source_records: list[dict[str, Any]],
    prediction_frame: pd.DataFrame,
    artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert OPS-SAT model output into database
    prediction records.
    """

    if len(source_records) != len(
        prediction_frame
    ):
        raise RuntimeError(
            "Model output count does not match "
            "the PostgreSQL feature count."
        )

    trained_channels = {
        str(channel)
        for channel in artifact.get(
            "channels",
            [],
        )
    }

    thresholds = _require_dictionary(
        artifact.get("thresholds"),
        "thresholds",
    )

    artifact_version = artifact.get(
        "artifact_version"
    )

    model_rows = prediction_frame.to_dict(
        orient="records"
    )

    prediction_records: list[
        dict[str, Any]
    ] = []

    for source_record, prediction_row in zip(
        source_records,
        model_rows,
        strict=True,
    ):
        sample_metadata = _require_dictionary(
            source_record.get(
                "sample_metadata"
            ),
            "sample_metadata",
        )

        channel = str(
            sample_metadata.get(
                "channel",
                prediction_row.get(
                    "channel",
                    "unknown",
                ),
            )
        )

        segment = sample_metadata.get(
            "segment",
            prediction_row.get(
                "segment"
            ),
        )

        risk_level = str(
            prediction_row[
                "risk_level"
            ]
        )

        predicted_anomaly = bool(
            int(
                prediction_row[
                    "prediction"
                ]
            )
        )

        feature_contributions_value = (
            prediction_row.get(
                "feature_contributions",
                {},
            )
        )

        if not isinstance(
            feature_contributions_value,
            dict,
        ):
            feature_contributions_value = {}

        isolation_score = float(
            prediction_row[
                "isolation_score"
            ]
        )

        supervised_score = float(
            prediction_row[
                "supervised_score"
            ]
        )

        hybrid_score = float(
            prediction_row[
                "hybrid_score"
            ]
        )

        confidence = float(
            prediction_row[
                "confidence"
            ]
        )

        decision_margin = float(
            prediction_row[
                "decision_margin"
            ]
        )

        top_feature = str(
            prediction_row[
                "top_feature"
            ]
        )

        top_feature_contribution = float(
            prediction_row[
                "top_feature_contribution"
            ]
        )

        out_of_distribution = (
            bool(trained_channels)
            and channel not in trained_channels
        )

        human_review_required = (
            risk_level in {
                "Watch",
                "Warning",
                "Critical",
            }
            or out_of_distribution
        )

        prediction_records.append(
            {
                "telemetry_sample_id": int(
                    source_record[
                        "telemetry_sample_id"
                    ]
                ),
                "predicted_anomaly": (
                    predicted_anomaly
                ),
                "risk_level": risk_level,
                "risk_score": hybrid_score,
                "confidence_score": confidence,
                "isolation_score": (
                    isolation_score
                ),
                "forecast_residual_score": None,

                # The supervised score is not a
                # rule-engine score, so it remains
                # inside prediction_metadata.
                "rule_score": None,

                "persistence_score": None,
                "early_warning_score": None,
                "top_feature": top_feature,
                "out_of_distribution": (
                    out_of_distribution
                ),
                "human_review_required": (
                    human_review_required
                ),
                "explanation": str(
                    prediction_row[
                        "explanation"
                    ]
                ),
                "feature_contributions": (
                    feature_contributions_value
                ),
                "rule_violations": [],
                "affected_subsystems": [
                    channel
                ],
                "prediction_metadata": {
                    "production_prediction": True,
                    "dataset_family": "OPS-SAT",
                    "record_level": (
                        "segment_features"
                    ),
                    "channel": channel,
                    "segment": segment,
                    "prediction_label": str(
                        prediction_row[
                            "prediction_label"
                        ]
                    ),
                    "supervised_score": (
                        supervised_score
                    ),
                    "decision_margin": (
                        decision_margin
                    ),
                    "top_feature_contribution": (
                        top_feature_contribution
                    ),
                    "thresholds": thresholds,
                    "artifact_version": (
                        artifact_version
                    ),
                },
            }
        )

    return prediction_records


def _severity_from_risk_level(
    risk_level: str,
) -> str:
    """
    Convert a prediction risk level into
    an incident severity.
    """

    if risk_level == "Critical":
        return "Critical"

    if risk_level == "Warning":
        return "Warning"

    return "Watch"


def _group_anomalous_predictions(
    source_records: list[dict[str, Any]],
    prediction_records: list[dict[str, Any]],
    max_sample_gap: int,
) -> list[
    list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ]
]:
    """
    Group nearby anomalous predictions on the same channel.

    The imported OPS-SAT feature dataset does not contain
    timestamps. Therefore, sample_index adjacency is used
    as a conservative grouping proxy.
    """

    if max_sample_gap < 1:
        raise ValueError(
            "max_sample_gap must be at least 1."
        )

    if len(source_records) != len(
        prediction_records
    ):
        raise RuntimeError(
            "Source-record and prediction counts "
            "do not match."
        )

    candidates: list[
        tuple[
            str,
            int,
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    for source_record, prediction_record in zip(
        source_records,
        prediction_records,
        strict=True,
    ):
        if not bool(
            prediction_record[
                "predicted_anomaly"
            ]
        ):
            continue

        sample_metadata = _require_dictionary(
            source_record.get(
                "sample_metadata"
            ),
            "sample_metadata",
        )

        channel = str(
            sample_metadata.get(
                "channel",
                "unknown",
            )
        )

        sample_index = int(
            source_record[
                "sample_index"
            ]
        )

        candidates.append(
            (
                channel,
                sample_index,
                source_record,
                prediction_record,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    groups: list[
        list[
            tuple[
                dict[str, Any],
                dict[str, Any],
            ]
        ]
    ] = []

    current_group: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    previous_channel: str | None = None
    previous_sample_index: int | None = None

    for (
        channel,
        sample_index,
        source_record,
        prediction_record,
    ) in candidates:
        starts_new_group = (
            not current_group
            or channel != previous_channel
            or previous_sample_index is None
            or (
                sample_index
                - previous_sample_index
                > max_sample_gap
            )
        )

        if starts_new_group:
            if current_group:
                groups.append(
                    current_group
                )

            current_group = []

        current_group.append(
            (
                source_record,
                prediction_record,
            )
        )

        previous_channel = channel
        previous_sample_index = sample_index

    if current_group:
        groups.append(
            current_group
        )

    return groups


def _highest_group_severity(
    group: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ],
) -> str:
    """
    Return the highest severity detected inside
    an incident group.
    """

    severities = [
        _severity_from_risk_level(
            str(
                prediction_record[
                    "risk_level"
                ]
            )
        )
        for _, prediction_record in group
    ]

    return max(
        severities,
        key=lambda severity: (
            INCIDENT_SEVERITY_ORDER[
                severity
            ]
        ),
    )



def _safe_upload_file_name(
    file_name: str,
) -> str:
    """
    Return a conservative file name with no directory traversal.
    """

    original_name = Path(file_name).name.strip()

    if not original_name:
        original_name = "uploaded_opssat.csv"

    safe_name = "".join(
        character
        if character.isalnum()
        or character in {"-", "_", "."}
        else "_"
        for character in original_name
    )

    if not safe_name.lower().endswith(".csv"):
        safe_name += ".csv"

    return safe_name[:160]


def _json_safe_value(
    value: object,
) -> object:
    """
    Convert pandas/numpy scalar values into JSON-safe values.
    """

    if value is None:
        return None

    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    item_method = getattr(
        value,
        "item",
        None,
    )

    if callable(item_method):
        try:
            value = item_method()
        except (TypeError, ValueError):
            pass

    if isinstance(
        value,
        (str, bool, int),
    ):
        return value

    if isinstance(value, float):
        if not pd.notna(value):
            return None

        if value in {
            float("inf"),
            float("-inf"),
        }:
            raise ValueError(
                "Infinite values cannot be stored in JSONB."
            )

        return value

    return str(value)


def _build_uploaded_feature_records(
    feature_frame: pd.DataFrame,
    numeric_features: list[str],
    source_file_name: str,
    upload_kind: str,
) -> list[dict[str, Any]]:
    """
    Convert validated uploaded features into telemetry records.
    """

    required_columns = {
        "segment",
        "channel",
        *numeric_features,
    }

    missing_columns = sorted(
        required_columns
        - set(feature_frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Uploaded feature frame is missing: "
            + ", ".join(missing_columns)
        )

    records: list[dict[str, Any]] = []

    for sample_index, (_, row) in enumerate(
        feature_frame.reset_index(drop=True).iterrows()
    ):
        feature_values: dict[str, float] = {}

        for feature_name in numeric_features:
            raw_value = row[feature_name]

            if pd.isna(raw_value):
                raise ValueError(
                    "Uploaded feature contains a missing value: "
                    f"row={sample_index}, feature={feature_name}"
                )

            numeric_value = float(raw_value)

            if numeric_value in {
                float("inf"),
                float("-inf"),
            }:
                raise ValueError(
                    "Uploaded feature contains an infinite value: "
                    f"row={sample_index}, feature={feature_name}"
                )

            feature_values[feature_name] = numeric_value

        segment = _json_safe_value(
            row.get("segment", sample_index)
        )

        channel = str(
            row.get("channel", "unknown")
        ).strip() or "unknown"

        ground_truth_label: bool | None = None

        if "anomaly" in feature_frame.columns:
            label_value = row.get("anomaly")

            if label_value is not None and pd.notna(label_value):
                ground_truth_label = bool(
                    int(float(label_value))
                )

        train_flag = _json_safe_value(
            row.get("train")
            if "train" in feature_frame.columns
            else None
        )

        records.append(
            {
                "sample_index": sample_index,
                "timestamp": None,
                "segment_identifier": str(segment),
                "split_type": "upload",
                "ground_truth_label": ground_truth_label,
                "anomaly_type": None,
                "sample_metadata": {
                    "segment": segment,
                    "channel": channel,
                    "train_flag": train_flag,
                    "source_file_name": source_file_name,
                    "upload_kind": upload_kind,
                    "original_feature_row": sample_index,
                },
                "feature_values": feature_values,
            }
        )

    if not records:
        raise ValueError(
            "Uploaded feature frame cannot be empty."
        )

    return records


def persist_uploaded_opssat_analysis(
    feature_frame: pd.DataFrame,
    original_file_bytes: bytes,
    original_file_name: str,
    model_version_id: UUID,
    artifact_path: Path,
    project_root: Path,
    upload_kind: str,
    validation_metadata: dict[str, Any] | None = None,
    max_incident_sample_gap: int = 1,
) -> UploadedOpsSatPersistenceResult:
    """
    Persist an uploaded OPS-SAT CSV and its complete analysis.

    The original CSV and normalized feature CSV are written to
    local project storage and registered in PostgreSQL. The
    validated feature rows are then stored as telemetry samples,
    analyzed by the real model, and converted into incidents.
    """

    if not original_file_bytes:
        raise ValueError(
            "Uploaded CSV bytes cannot be empty."
        )

    if feature_frame.empty:
        raise ValueError(
            "Uploaded feature frame cannot be empty."
        )

    resolved_project_root = (
        project_root.expanduser().resolve()
    )

    resolved_artifact_path = (
        artifact_path.expanduser().resolve()
    )

    artifact = load_artifact(
        resolved_artifact_path
    )

    numeric_features = [
        str(feature_name)
        for feature_name in artifact[
            "numeric_features"
        ]
    ]

    clean_file_name = _safe_upload_file_name(
        original_file_name
    )

    content_hash = sha256(
        original_file_bytes
    ).hexdigest()

    created_at = datetime.now(
        timezone.utc
    )

    upload_token = (
        created_at.strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + content_hash[:10]
        + "-"
        + uuid4().hex[:6]
    )

    dataset_code = (
        "UPLOAD-" + upload_token.upper()
    )

    upload_directory = (
        resolved_project_root
        / "data"
        / "opssat"
        / "uploads"
        / upload_token
    )

    upload_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    original_file_path = (
        upload_directory
        / clean_file_name
    )

    processed_file_path = (
        upload_directory
        / "processed_features.csv"
    )

    original_file_path.write_bytes(
        original_file_bytes
    )

    feature_frame.to_csv(
        processed_file_path,
        index=False,
    )

    validation = dict(
        validation_metadata or {}
    )

    validation_messages = [
        str(message)
        for message in validation.get(
            "messages",
            [],
        )
    ]

    removed_rows = int(
        validation.get(
            "removed_rows",
            0,
        )
        or 0
    )

    validation_status = (
        "warning"
        if removed_rows > 0
        or validation_messages
        else "valid"
    )

    label_coverage = float(
        validation.get(
            "label_coverage",
            0.0,
        )
        or 0.0
    )

    original_relative_path = (
        original_file_path
        .relative_to(resolved_project_root)
        .as_posix()
    )

    processed_relative_path = (
        processed_file_path
        .relative_to(resolved_project_root)
        .as_posix()
    )

    dataset_id = create_dataset(
        name=(
            "Uploaded OPS-SAT — "
            + clean_file_name
        ),
        dataset_code=dataset_code,
        source_type="upload",
        source_organization="User Upload",
        description=(
            "User-supplied OPS-SAT-compatible CSV "
            "validated and persisted by MissionGuard AI."
        ),
        version=created_at.isoformat(),
        row_count=len(feature_frame),
        feature_count=len(numeric_features),
        is_labeled=label_coverage > 0.0,
        metadata={
            "dataset_family": "OPS-SAT",
            "upload_kind": upload_kind,
            "sha256_hash": content_hash,
            "original_file_name": clean_file_name,
            "original_file_path": original_relative_path,
            "processed_file_path": processed_relative_path,
            "validation": validation,
        },
    )

    original_dataset_file_id = create_dataset_file(
        dataset_id=dataset_id,
        file_name=clean_file_name,
        file_role="raw",
        file_path=original_relative_path,
        storage_provider="local",
        file_size_bytes=len(original_file_bytes),
        mime_type="text/csv",
        sha256_hash=content_hash,
        row_count=int(
            validation.get(
                "input_rows",
                len(feature_frame),
            )
            or len(feature_frame)
        ),
        metadata={
            "upload_kind": upload_kind,
            "original_upload": True,
        },
    )

    processed_bytes = processed_file_path.read_bytes()

    processed_dataset_file_id = create_dataset_file(
        dataset_id=dataset_id,
        file_name=processed_file_path.name,
        file_role="processed",
        file_path=processed_relative_path,
        storage_provider="local",
        file_size_bytes=len(processed_bytes),
        mime_type="text/csv",
        sha256_hash=sha256(
            processed_bytes
        ).hexdigest(),
        row_count=len(feature_frame),
        metadata={
            "feature_schema_name": (
                "opssat_segment_features"
            ),
            "feature_schema_version": "1.0",
            "numeric_features": numeric_features,
        },
    )

    telemetry_session_id = create_telemetry_session(
        session_name=(
            "Uploaded OPS-SAT — "
            + clean_file_name
            + " — "
            + created_at.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        ),
        source_type="uploaded_csv",
        dataset_id=dataset_id,
        source_file_name=clean_file_name,
        total_samples=len(feature_frame),
        validation_status=validation_status,
        metadata={
            "dataset_family": "OPS-SAT",
            "upload_kind": upload_kind,
            "sha256_hash": content_hash,
            "raw_file_path": original_relative_path,
            "processed_file_path": processed_relative_path,
            "validation": validation,
        },
    )

    telemetry_records = (
        _build_uploaded_feature_records(
            feature_frame=feature_frame,
            numeric_features=numeric_features,
            source_file_name=clean_file_name,
            upload_kind=upload_kind,
        )
    )

    missing_value_summary = {
        str(column): int(count)
        for column, count in (
            feature_frame
            .isna()
            .sum()
            .items()
        )
        if int(count) > 0
    }

    save_telemetry_batch(
        session_id=telemetry_session_id,
        records=telemetry_records,
        feature_schema_name=(
            "opssat_segment_features"
        ),
        feature_schema_version="1.0",
        quality_report={
            "row_count": len(feature_frame),
            "invalid_timestamps": 0,
            "duplicate_timestamps": 0,
            "long_missing_gaps": 0,
            "constant_sensors": [],
            "out_of_domain_values": {},
            "missing_value_summary": (
                missing_value_summary
            ),
            "sampling_report": {
                "record_level": "segment_features",
                "timestamps_available": False,
            },
            "validation_messages": (
                validation_messages
            ),
            "overall_status": validation_status,
        },
    )

    analysis_result = run_real_opssat_analysis(
        telemetry_session_id=(
            telemetry_session_id
        ),
        model_version_id=model_version_id,
        artifact_path=resolved_artifact_path,
        max_incident_sample_gap=(
            max_incident_sample_gap
        ),
        run_type="live_analysis",
        run_metadata={
            "source_type": "uploaded_csv",
            "dataset_id": str(dataset_id),
            "source_file_name": clean_file_name,
            "upload_kind": upload_kind,
            "sha256_hash": content_hash,
            "raw_file_path": original_relative_path,
            "processed_file_path": processed_relative_path,
        },
    )

    return UploadedOpsSatPersistenceResult(
        dataset_id=dataset_id,
        original_dataset_file_id=(
            original_dataset_file_id
        ),
        processed_dataset_file_id=(
            processed_dataset_file_id
        ),
        telemetry_session_id=(
            telemetry_session_id
        ),
        analysis_result=analysis_result,
        original_file_path=(
            original_file_path
        ),
        processed_file_path=(
            processed_file_path
        ),
        sha256_hash=content_hash,
    )

def run_real_opssat_analysis(
    telemetry_session_id: UUID,
    model_version_id: UUID,
    artifact_path: Path,
    max_incident_sample_gap: int = 1,
    run_type: str = "benchmark",
    run_metadata: dict[str, Any] | None = None,
) -> OpsSatAnalysisResult:
    """
    Run the real OPS-SAT Hybrid model and persist results.

    Workflow:

    1. Read feature vectors from PostgreSQL.
    2. Load the trusted local model artifact.
    3. Run real model inference.
    4. Store all predictions.
    5. Group nearby anomaly predictions into incidents.
    6. Complete the analysis run.
    """

    resolved_artifact_path = (
        artifact_path
        .expanduser()
        .resolve()
    )

    if not resolved_artifact_path.exists():
        raise FileNotFoundError(
            "OPS-SAT artifact was not found: "
            f"{resolved_artifact_path}"
        )

    artifact = load_artifact(
        resolved_artifact_path
    )

    numeric_features = [
        str(feature_name)
        for feature_name in artifact[
            "numeric_features"
        ]
    ]

    source_records = (
        list_session_feature_records(
            session_id=telemetry_session_id,
            schema_name=(
                "opssat_segment_features"
            ),
        )
    )

    if not source_records:
        raise RuntimeError(
            "No OPS-SAT feature vectors were found "
            "for the selected telemetry session."
        )

    inference_frame = _build_inference_frame(
        records=source_records,
        numeric_features=numeric_features,
    )

    base_run_metadata: dict[str, Any] = {
        "production_run": True,
        "dataset_family": "OPS-SAT",
        "record_level": "segment_features",
        "artifact_path": str(
            resolved_artifact_path
        ),
        "input_rows": len(
            inference_frame
        ),
        "ground_truth_used_for_inference": False,
        "incident_grouping_enabled": True,
        "max_incident_sample_gap": (
            max_incident_sample_gap
        ),
    }

    if run_metadata:
        base_run_metadata.update(
            run_metadata
        )

    analysis_run_id = create_analysis_run(
        session_id=telemetry_session_id,
        model_version_id=model_version_id,
        run_type=run_type,
        status="running",
        metadata=base_run_metadata,
    )

    try:
        prediction_frame = (
            predict_feature_rows(
                inference_frame,
                artifact,
            )
        )

        prediction_records = (
            _build_prediction_records(
                source_records=source_records,
                prediction_frame=prediction_frame,
                artifact=artifact,
            )
        )

        prediction_result = (
            save_prediction_batch(
                analysis_run_id=analysis_run_id,
                prediction_records=prediction_records,
            )
        )

        incident_groups = (
            _group_anomalous_predictions(
                source_records=source_records,
                prediction_records=(
                    prediction_records
                ),
                max_sample_gap=(
                    max_incident_sample_gap
                ),
            )
        )

        mission_health_incidents: list[
            dict[str, object]
        ] = []

        for group_number, group in enumerate(
            incident_groups,
            start=1,
        ):
            start_source_record = (
                group[0][0]
            )

            end_source_record = (
                group[-1][0]
            )

            (
                peak_source_record,
                peak_prediction_record,
            ) = max(
                group,
                key=lambda item: float(
                    item[1][
                        "risk_score"
                    ]
                ),
            )

            start_metadata = (
                _require_dictionary(
                    start_source_record.get(
                        "sample_metadata"
                    ),
                    "sample_metadata",
                )
            )

            end_metadata = (
                _require_dictionary(
                    end_source_record.get(
                        "sample_metadata"
                    ),
                    "sample_metadata",
                )
            )

            peak_metadata = (
                _require_dictionary(
                    peak_source_record.get(
                        "sample_metadata"
                    ),
                    "sample_metadata",
                )
            )

            channel = str(
                peak_metadata.get(
                    "channel",
                    "unknown",
                )
            )

            start_segment = (
                start_metadata.get(
                    "segment",
                    start_source_record[
                        "sample_index"
                    ],
                )
            )

            end_segment = (
                end_metadata.get(
                    "segment",
                    end_source_record[
                        "sample_index"
                    ],
                )
            )

            sample_indexes = [
                int(
                    source_record[
                        "sample_index"
                    ]
                )
                for source_record, _ in group
            ]

            segment_ids = [
                _require_dictionary(
                    source_record.get(
                        "sample_metadata"
                    ),
                    "sample_metadata",
                ).get(
                    "segment",
                    source_record[
                        "sample_index"
                    ],
                )
                for source_record, _ in group
            ]

            severity = (
                _highest_group_severity(
                    group
                )
            )

            peak_risk_score = float(
                peak_prediction_record[
                    "risk_score"
                ]
            )

            peak_confidence = max(
                float(
                    prediction_record[
                        "confidence_score"
                    ]
                )
                for _, prediction_record in group
            )

            human_review_required = any(
                bool(
                    prediction_record[
                        "human_review_required"
                    ]
                )
                for _, prediction_record in group
            )

            incident_code = (
                "OPS-"
                + _safe_incident_token(
                    channel
                )
                + "-"
                + analysis_run_id.hex[:8].upper()
                + f"-{group_number:03d}"
            )

            _ = create_incident(
                analysis_run_id=analysis_run_id,
                incident_code=incident_code,
                start_sample_id=int(
                    start_source_record[
                        "telemetry_sample_id"
                    ]
                ),
                end_sample_id=int(
                    end_source_record[
                        "telemetry_sample_id"
                    ]
                ),
                started_at=(
                    start_source_record.get(
                        "timestamp"
                    )
                ),
                ended_at=(
                    end_source_record.get(
                        "timestamp"
                    )
                ),
                duration_samples=len(
                    group
                ),
                severity=severity,
                peak_risk_score=(
                    peak_risk_score
                ),
                peak_confidence=(
                    peak_confidence
                ),
                top_feature=str(
                    peak_prediction_record.get(
                        "top_feature"
                    )
                    or "unknown"
                ),
                affected_subsystems=[
                    channel
                ],
                human_review_required=(
                    human_review_required
                ),
                status="open",
                summary=(
                    f"{severity} OPS-SAT incident "
                    f"on channel {channel}. "
                    f"{len(group)} nearby anomalous "
                    f"segment(s) were grouped. "
                    f"Peak hybrid risk: "
                    f"{peak_risk_score:.2f}/100."
                ),
                metadata={
                    "production_incident": True,
                    "dataset_family": "OPS-SAT",
                    "record_level": (
                        "segment_features"
                    ),
                    "channel": channel,
                    "start_segment": (
                        start_segment
                    ),
                    "end_segment": (
                        end_segment
                    ),
                    "segment_ids": (
                        segment_ids
                    ),
                    "sample_indexes": (
                        sample_indexes
                    ),
                    "peak_sample_index": int(
                        peak_source_record[
                            "sample_index"
                        ]
                    ),
                    "grouping_strategy": (
                        "same_channel_nearby_"
                        "sample_indexes"
                    ),
                    "max_sample_gap": (
                        max_incident_sample_gap
                    ),
                },
            )

            mission_health_incidents.append(
                {
                    "severity": severity,
                    "status": "open",
                }
            )

        total_incidents = len(
            incident_groups
        )

        hybrid_scores = (
            prediction_frame[
                "hybrid_score"
            ]
            .astype(float)
            .tolist()
        )

        mean_risk_score = float(
            sum(hybrid_scores)
            / max(
                len(hybrid_scores),
                1,
            )
        )

        maximum_risk_score = float(
            max(
                hybrid_scores,
                default=0.0,
            )
        )

        anomaly_rate = (
            prediction_result.anomaly_predictions
            / max(
                prediction_result.stored_predictions,
                1,
            )
        )

        mission_health = calculate_mission_health_score(
            frame=prediction_frame,
            incidents=mission_health_incidents,
        )

        complete_analysis_run(
            analysis_run_id=analysis_run_id,
            total_predictions=(
                prediction_result
                .stored_predictions
            ),
            total_anomalies=(
                prediction_result
                .anomaly_predictions
            ),
            total_incidents=(
                total_incidents
            ),
            mission_health_score=float(
                mission_health["score"]
            ),
            metadata={
                **base_run_metadata,
                "total_input_rows": len(
                    inference_frame
                ),
                "anomaly_rate": anomaly_rate,
                "mean_hybrid_score": (
                    mean_risk_score
                ),
                "maximum_hybrid_score": (
                    maximum_risk_score
                ),
                "incident_strategy": (
                    "same_channel_nearby_"
                    "sample_indexes"
                ),
                "max_incident_sample_gap": (
                    max_incident_sample_gap
                ),
                "ground_truth_used_for_inference": False,
                "mission_health": mission_health,
            },
        )

    except Exception as error:
        fail_analysis_run(
            analysis_run_id=analysis_run_id,
            error_message=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )

        raise

    return OpsSatAnalysisResult(
        analysis_run_id=analysis_run_id,
        total_predictions=(
            prediction_result
            .stored_predictions
        ),
        total_anomalies=(
            prediction_result
            .anomaly_predictions
        ),
        total_incidents=(
            total_incidents
        ),
        mean_risk_score=(
            mean_risk_score
        ),
        maximum_risk_score=(
            maximum_risk_score
        ),
        mission_health_score=float(
            mission_health["score"]
        ),
        mission_health_status=str(
            mission_health["status"]
        ),
    )