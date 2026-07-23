from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from database.repositories.datasets import (
    create_dataset,
    create_dataset_file,
)
from database.repositories.telemetry import (
    save_telemetry_batch,
)
from database.repositories.telemetry_sessions import (
    create_telemetry_session,
)

FEATURE_COLUMNS: tuple[str, ...] = (
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
)


METADATA_COLUMNS: tuple[str, ...] = (
    "segment",
    "anomaly",
    "train",
    "channel",
)


REQUIRED_COLUMNS: set[str] = (
    set(FEATURE_COLUMNS)
    | set(METADATA_COLUMNS)
)


@dataclass(frozen=True)
class OpsSatImportResult:
    """
    Summary of an OPS-SAT CSV import operation.
    """

    mission_id: UUID
    dataset_id: UUID
    dataset_file_id: UUID
    telemetry_session_id: UUID
    quality_report_id: UUID | None
    row_count: int
    feature_count: int
    stored_samples: int
    stored_feature_vectors: int


def _calculate_sha256(
    file_path: Path,
) -> str:
    """
    Calculate a file SHA-256 hash without loading
    the entire file into memory.
    """

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def _to_python_value(
    value: object,
) -> Any:
    """
    Convert pandas and NumPy scalar values into
    standard Python values suitable for JSONB.
    """

    if pd.isna(value):
        return None

    item_method = getattr(
        value,
        "item",
        None,
    )

    if callable(item_method):
        return item_method()

    return value


def _validate_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate that the OPS-SAT feature file contains
    the columns required by MissionGuard.
    """

    if dataframe.empty:
        raise ValueError(
            "The OPS-SAT CSV file contains no rows."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        missing_list = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "The OPS-SAT CSV file is missing "
            f"required columns: {missing_list}"
        )

    anomaly_values = pd.to_numeric(
        dataframe["anomaly"],
        errors="coerce",
    )

    if anomaly_values.isna().any():
        raise ValueError(
            "The anomaly column contains invalid "
            "or missing values."
        )

    unique_labels = set(
        float(value)
        for value in anomaly_values.unique()
    )

    valid_labels = {
        0.0,
        1.0,
    }

    if not unique_labels.issubset(
        valid_labels
    ):
        raise ValueError(
            "The anomaly column must contain "
            "only 0 or 1."
        )


def _build_telemetry_records(
    dataframe: pd.DataFrame,
    source_file_name: str,
) -> list[dict[str, Any]]:
    """
    Convert OPS-SAT feature rows into records accepted
    by save_telemetry_batch().
    """

    records: list[dict[str, Any]] = []

    row_records = dataframe.to_dict(
        orient="records"
    )

    for sample_index, row in enumerate(
        row_records
    ):
        segment_value = _to_python_value(
            row["segment"]
        )

        channel_value = str(
            row["channel"]
        ).strip()

        train_value = _to_python_value(
            row["train"]
        )

        anomaly_value = float(
            row["anomaly"]
        )

        is_anomaly = (
            anomaly_value == 1.0
        )

        feature_values = {
            column_name: _to_python_value(
                row[column_name]
            )
            for column_name in FEATURE_COLUMNS
        }

        records.append(
            {
                "sample_index": sample_index,
                "timestamp": None,
                "segment_identifier": (
                    f"{channel_value}:"
                    f"{segment_value}"
                ),
                "split_type": "validation",
                "ground_truth_label": (
                    is_anomaly
                ),
                "anomaly_type": (
                    "opssat_segment_anomaly"
                    if is_anomaly
                    else None
                ),
                "sample_metadata": {
                    "segment": segment_value,
                    "channel": channel_value,
                    "train_flag": train_value,
                    "source_file": (
                        source_file_name
                    ),
                    "record_level": (
                        "segment_features"
                    ),
                    "timestamp_available": False,
                },
                "feature_values": (
                    feature_values
                ),
            }
        )

    return records


def _build_quality_report(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a quality report from the imported feature file.
    """

    missing_counts = (
        dataframe
        .isna()
        .sum()
    )

    missing_value_summary = {
        str(column_name): int(count)
        for column_name, count
        in missing_counts.items()
        if int(count) > 0
    }

    duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    constant_features = [
        column_name
        for column_name in FEATURE_COLUMNS
        if dataframe[
            column_name
        ].nunique(
            dropna=False
        ) <= 1
    ]

    sampling_values = sorted(
        int(value)
        for value in dataframe[
            "sampling"
        ].dropna().unique()
    )

    validation_messages: list[str] = [
        (
            "The imported file contains segment-level "
            "features rather than timestamp-level "
            "raw telemetry."
        ),
        (
            "No timestamp column is available, so "
            "telemetry sample timestamps were stored "
            "as NULL."
        ),
    ]

    if duplicate_rows > 0:
        validation_messages.append(
            f"{duplicate_rows} duplicate rows "
            "were detected."
        )

    if missing_value_summary:
        validation_messages.append(
            "Missing feature values were detected."
        )

    if constant_features:
        validation_messages.append(
            "Constant feature columns were detected: "
            + ", ".join(constant_features)
        )

    overall_status = "valid"

    if (
        duplicate_rows > 0
        or missing_value_summary
    ):
        overall_status = "warning"

    return {
        "row_count": len(dataframe),
        "invalid_timestamps": 0,
        "duplicate_timestamps": 0,
        "long_missing_gaps": 0,
        "constant_sensors": (
            constant_features
        ),
        "out_of_domain_values": {},
        "missing_value_summary": (
            missing_value_summary
        ),
        "sampling_report": {
            "timestamp_available": False,
            "sampling_values": (
                sampling_values
            ),
            "record_level": (
                "segment_features"
            ),
        },
        "validation_messages": (
            validation_messages
        ),
        "overall_status": (
            overall_status
        ),
    }


def import_opssat_feature_csv(
    mission_id: UUID,
    csv_file_path: Path,
    dataset_name: str = (
        "OPS-SAT Validation Features"
    ),
    dataset_code: str | None = None,
) -> OpsSatImportResult:
    """
    Import a processed OPS-SAT segment-feature CSV file.

    The operation creates:

    1. Dataset
    2. Dataset file
    3. Telemetry session
    4. Telemetry samples
    5. Feature vectors
    6. Data-quality report
    """

    file_path = (
        csv_file_path
        .expanduser()
        .resolve()
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"OPS-SAT CSV file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            "The OPS-SAT input file must be CSV."
        )

    dataframe = pd.read_csv(
        file_path,
        low_memory=False,
    )

    _validate_dataframe(
        dataframe
    )

    generated_code = (
        dataset_code
        or (
            "OPS-VAL-"
            + uuid4().hex[:8].upper()
        )
    )

    row_count = len(
        dataframe
    )

    feature_count = len(
        FEATURE_COLUMNS
    )

    dataset_id = create_dataset(
        name=dataset_name,
        dataset_code=generated_code,
        source_type="opssat",
        source_organization=(
            "European Space Agency"
        ),
        description=(
            "Processed OPS-SAT segment-level "
            "validation features imported into "
            "MissionGuard AI."
        ),
        version="1.0",
        row_count=row_count,
        feature_count=feature_count,
        is_labeled=True,
        metadata={
            "record_level": (
                "segment_features"
            ),
            "timestamp_available": False,
            "label_column": "anomaly",
            "channel_column": "channel",
            "segment_column": "segment",
            "feature_columns": list(
                FEATURE_COLUMNS
            ),
        },
    )

    dataset_file_id = create_dataset_file(
        dataset_id=dataset_id,
        file_name=file_path.name,
        file_role="processed",
        file_path=str(file_path),
        storage_provider="local",
        file_size_bytes=(
            file_path.stat().st_size
        ),
        mime_type="text/csv",
        sha256_hash=_calculate_sha256(
            file_path
        ),
        row_count=row_count,
        metadata={
            "dataset_family": "opssat",
            "record_level": (
                "segment_features"
            ),
        },
    )

    telemetry_session_id = (
        create_telemetry_session(
            session_name=(
                "OPS-SAT Validation Feature Import"
            ),
            source_type="opssat",
            mission_id=mission_id,
            dataset_id=dataset_id,
            source_file_name=(
                file_path.name
            ),
            sampling_interval_seconds=None,
            total_samples=row_count,
            validation_status="valid",
            metadata={
                "record_level": (
                    "segment_features"
                ),
                "timestamp_available": False,
            },
        )
    )

    records = _build_telemetry_records(
        dataframe=dataframe,
        source_file_name=file_path.name,
    )

    quality_report = _build_quality_report(
        dataframe
    )

    save_result = save_telemetry_batch(
        session_id=telemetry_session_id,
        records=records,
        feature_schema_name=(
            "opssat_segment_features"
        ),
        feature_schema_version="1.0",
        quality_report=quality_report,
    )

    return OpsSatImportResult(
        mission_id=mission_id,
        dataset_id=dataset_id,
        dataset_file_id=dataset_file_id,
        telemetry_session_id=(
            telemetry_session_id
        ),
        quality_report_id=(
            save_result.quality_report_id
        ),
        row_count=row_count,
        feature_count=feature_count,
        stored_samples=(
            save_result.inserted_samples
        ),
        stored_feature_vectors=(
            save_result
            .inserted_feature_vectors
            
        ),
    )