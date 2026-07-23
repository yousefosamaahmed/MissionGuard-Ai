from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import database_session
from database.tables import get_table


@dataclass(frozen=True)
class TelemetrySaveResult:
    """
    Summary returned after saving telemetry data.
    """

    inserted_samples: int
    inserted_feature_vectors: int
    quality_report_id: UUID | None
    sample_ids_by_index: dict[int, int]


def _to_uuid(value: object) -> UUID:
    """
    Convert a PostgreSQL UUID value to Python UUID.
    """

    if isinstance(value, UUID):
        return value

    return UUID(str(value))


def _validate_telemetry_records(
    records: list[dict[str, Any]],
) -> None:
    """
    Validate telemetry records before database insertion.
    """

    if not records:
        raise ValueError(
            "Telemetry records cannot be empty."
        )

    sample_indexes: set[int] = set()

    for position, record in enumerate(records):
        if "sample_index" not in record:
            raise ValueError(
                f"Record {position} is missing sample_index."
            )

        if "feature_values" not in record:
            raise ValueError(
                f"Record {position} is missing feature_values."
            )

        sample_index = record["sample_index"]

        if not isinstance(sample_index, int):
            raise TypeError(
                "sample_index must be an integer."
            )

        if sample_index < 0:
            raise ValueError(
                "sample_index cannot be negative."
            )

        if sample_index in sample_indexes:
            raise ValueError(
                f"Duplicate sample_index: {sample_index}"
            )

        sample_indexes.add(sample_index)

        feature_values = record["feature_values"]

        if not isinstance(feature_values, dict):
            raise TypeError(
                "feature_values must be a dictionary."
            )

        if not feature_values:
            raise ValueError(
                f"Feature values are empty for sample "
                f"{sample_index}."
            )


def save_telemetry_batch(
    session_id: UUID,
    records: list[dict[str, Any]],
    feature_schema_name: str = "missionguard_core",
    feature_schema_version: str = "1.0",
    quality_report: dict[str, Any] | None = None,
) -> TelemetrySaveResult:
    """
    Save telemetry samples, feature vectors and an optional
    data-quality report inside one database transaction.

    If the same session_id and sample_index already exist,
    the existing sample is updated instead of duplicated.
    """

    _validate_telemetry_records(records)

    clean_schema_name = feature_schema_name.strip()
    clean_schema_version = feature_schema_version.strip()

    if not clean_schema_name:
        raise ValueError(
            "Feature schema name cannot be empty."
        )

    if not clean_schema_version:
        raise ValueError(
            "Feature schema version cannot be empty."
        )

    telemetry_samples_table = get_table(
        "telemetry_samples"
    )

    feature_vectors_table = get_table(
        "feature_vectors"
    )

    quality_reports_table = get_table(
        "data_quality_reports"
    )

    sample_rows: list[dict[str, Any]] = []

    for record in records:
        sample_rows.append(
            {
                "session_id": session_id,
                "mission_phase_id": record.get(
                    "mission_phase_id"
                ),
                "sample_index": record["sample_index"],
                "timestamp": record.get("timestamp"),
                "segment_identifier": record.get(
                    "segment_identifier"
                ),
                "split_type": record.get(
                    "split_type",
                    "upload",
                ),
                "ground_truth_label": record.get(
                    "ground_truth_label"
                ),
                "anomaly_type": record.get(
                    "anomaly_type"
                ),
                "sample_metadata": record.get(
                    "sample_metadata",
                    {},
                ),
            }
        )

    sample_insert = pg_insert(
        telemetry_samples_table
    )

    sample_statement = (
        sample_insert
        .values(sample_rows)
        .on_conflict_do_update(
            constraint="uq_telemetry_sample_index",
            set_={
                "mission_phase_id": (
                    sample_insert
                    .excluded
                    .mission_phase_id
                ),
                "timestamp": (
                    sample_insert
                    .excluded
                    .timestamp
                ),
                "segment_identifier": (
                    sample_insert
                    .excluded
                    .segment_identifier
                ),
                "split_type": (
                    sample_insert
                    .excluded
                    .split_type
                ),
                "ground_truth_label": (
                    sample_insert
                    .excluded
                    .ground_truth_label
                ),
                "anomaly_type": (
                    sample_insert
                    .excluded
                    .anomaly_type
                ),
                "sample_metadata": (
                    sample_insert
                    .excluded
                    .sample_metadata
                ),
            },
        )
        .returning(
            telemetry_samples_table.c.id,
            telemetry_samples_table.c.sample_index,
        )
    )

    quality_report_id: UUID | None = None

    with database_session() as session:
        sample_result = session.execute(
            sample_statement
        )

        stored_sample_rows = (
            sample_result
            .mappings()
            .all()
        )

        if len(stored_sample_rows) != len(records):
            raise RuntimeError(
                "The number of stored telemetry samples "
                "does not match the input records."
            )

        sample_ids_by_index: dict[int, int] = {}

        for stored_sample in stored_sample_rows:
            sample_index = int(
                stored_sample["sample_index"]
            )

            sample_id = int(
                stored_sample["id"]
            )

            sample_ids_by_index[
                sample_index
            ] = sample_id

        feature_rows: list[dict[str, Any]] = []

        for record in records:
            sample_index = record["sample_index"]

            telemetry_sample_id = (
                sample_ids_by_index[
                    sample_index
                ]
            )

            feature_rows.append(
                {
                    "telemetry_sample_id": (
                        telemetry_sample_id
                    ),
                    "schema_name": (
                        clean_schema_name
                    ),
                    "schema_version": (
                        clean_schema_version
                    ),
                    "feature_values": record[
                        "feature_values"
                    ],
                }
            )

        feature_insert = pg_insert(
            feature_vectors_table
        )

        feature_statement = (
            feature_insert
            .values(feature_rows)
            .on_conflict_do_update(
                constraint=(
                    "uq_sample_feature_schema"
                ),
                set_={
                    "feature_values": (
                        feature_insert
                        .excluded
                        .feature_values
                    ),
                },
            )
            .returning(
                feature_vectors_table.c.id
            )
        )

        feature_result = session.execute(
            feature_statement
        )

        stored_feature_ids = (
            feature_result
            .scalars()
            .all()
        )

        if len(stored_feature_ids) != len(records):
            raise RuntimeError(
                "The number of stored feature vectors "
                "does not match the telemetry samples."
            )

        if quality_report is not None:
            report_values = {
                "session_id": session_id,
                "row_count": quality_report.get(
                    "row_count",
                    len(records),
                ),
                "invalid_timestamps": (
                    quality_report.get(
                        "invalid_timestamps",
                        0,
                    )
                ),
                "duplicate_timestamps": (
                    quality_report.get(
                        "duplicate_timestamps",
                        0,
                    )
                ),
                "long_missing_gaps": (
                    quality_report.get(
                        "long_missing_gaps",
                        0,
                    )
                ),
                "constant_sensors": (
                    quality_report.get(
                        "constant_sensors",
                        [],
                    )
                ),
                "out_of_domain_values": (
                    quality_report.get(
                        "out_of_domain_values",
                        {},
                    )
                ),
                "missing_value_summary": (
                    quality_report.get(
                        "missing_value_summary",
                        {},
                    )
                ),
                "sampling_report": (
                    quality_report.get(
                        "sampling_report",
                        {},
                    )
                ),
                "validation_messages": (
                    quality_report.get(
                        "validation_messages",
                        [],
                    )
                ),
                "overall_status": (
                    quality_report.get(
                        "overall_status",
                        "valid",
                    )
                ),
            }

            report_insert = pg_insert(
                quality_reports_table
            )

            report_statement = (
                report_insert
                .values(report_values)
                .on_conflict_do_update(
                    constraint=(
                        "uq_data_quality_session"
                    ),
                    set_={
                        "row_count": (
                            report_insert
                            .excluded
                            .row_count
                        ),
                        "invalid_timestamps": (
                            report_insert
                            .excluded
                            .invalid_timestamps
                        ),
                        "duplicate_timestamps": (
                            report_insert
                            .excluded
                            .duplicate_timestamps
                        ),
                        "long_missing_gaps": (
                            report_insert
                            .excluded
                            .long_missing_gaps
                        ),
                        "constant_sensors": (
                            report_insert
                            .excluded
                            .constant_sensors
                        ),
                        "out_of_domain_values": (
                            report_insert
                            .excluded
                            .out_of_domain_values
                        ),
                        "missing_value_summary": (
                            report_insert
                            .excluded
                            .missing_value_summary
                        ),
                        "sampling_report": (
                            report_insert
                            .excluded
                            .sampling_report
                        ),
                        "validation_messages": (
                            report_insert
                            .excluded
                            .validation_messages
                        ),
                        "overall_status": (
                            report_insert
                            .excluded
                            .overall_status
                        ),
                    },
                )
                .returning(
                    quality_reports_table.c.id
                )
            )

            stored_report_id = (
                session.execute(
                    report_statement
                )
                .scalar_one()
            )

            quality_report_id = _to_uuid(
                stored_report_id
            )

    return TelemetrySaveResult(
        inserted_samples=len(
            sample_ids_by_index
        ),
        inserted_feature_vectors=len(
            stored_feature_ids
        ),
        quality_report_id=quality_report_id,
        sample_ids_by_index=sample_ids_by_index,
    )


def list_telemetry_samples(
    session_id: UUID,
) -> list[dict[str, Any]]:
    """
    Return telemetry samples for one session.
    """

    telemetry_samples_table = get_table(
        "telemetry_samples"
    )

    statement = (
        select(telemetry_samples_table)
        .where(
            telemetry_samples_table.c.session_id
            == session_id
        )
        .order_by(
            telemetry_samples_table
            .c
            .sample_index
            .asc()
        )
    )

    with database_session() as session:
        rows = (
            session.execute(statement)
            .mappings()
            .all()
        )

    return [
        dict(row)
        for row in rows
    ]


def list_feature_vectors(
    session_id: UUID,
) -> list[dict[str, Any]]:
    """
    Return feature vectors belonging to a telemetry session.
    """

    telemetry_samples_table = get_table(
        "telemetry_samples"
    )

    feature_vectors_table = get_table(
        "feature_vectors"
    )

    statement = (
        select(
            feature_vectors_table
        )
        .join(
            telemetry_samples_table,
            feature_vectors_table
            .c
            .telemetry_sample_id
            == telemetry_samples_table.c.id,
        )
        .where(
            telemetry_samples_table.c.session_id
            == session_id
        )
        .order_by(
            telemetry_samples_table
            .c
            .sample_index
            .asc()
        )
    )

    with database_session() as session:
        rows = (
            session.execute(statement)
            .mappings()
            .all()
        )

    return [
        dict(row)
        for row in rows
    ]


def get_data_quality_report(
    session_id: UUID,
) -> dict[str, Any] | None:
    """
    Return the data-quality report for a session.
    """

    quality_reports_table = get_table(
        "data_quality_reports"
    )

    statement = (
        select(quality_reports_table)
        .where(
            quality_reports_table.c.session_id
            == session_id
        )
    )

    with database_session() as session:
        row = (
            session.execute(statement)
            .mappings()
            .one_or_none()
        )

    if row is None:
        return None

    return dict(row)

def list_session_feature_records(
    session_id: UUID,
    schema_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return telemetry samples joined with their feature vectors.

    Results are ordered by sample_index to preserve
    the original telemetry sequence.
    """

    telemetry_samples = get_table(
        "telemetry_samples"
    )

    feature_vectors = get_table(
        "feature_vectors"
    )

    statement = (
        select(
            telemetry_samples.c.id.label(
                "telemetry_sample_id"
            ),
            telemetry_samples.c.sample_index,
            telemetry_samples.c.timestamp,
            telemetry_samples.c.segment_identifier,
            telemetry_samples.c.ground_truth_label,
            telemetry_samples.c.anomaly_type,
            telemetry_samples.c.sample_metadata,
            feature_vectors.c.schema_name,
            feature_vectors.c.schema_version,
            feature_vectors.c.feature_values,
        )
        .join(
            feature_vectors,
            feature_vectors.c.telemetry_sample_id
            == telemetry_samples.c.id,
        )
        .where(
            telemetry_samples.c.session_id
            == session_id
        )
    )

    if schema_name is not None:
        statement = statement.where(
            feature_vectors.c.schema_name
            == schema_name
        )

    statement = statement.order_by(
        telemetry_samples.c.sample_index.asc()
    )

    with database_session() as session:
        rows = (
            session.execute(statement)
            .mappings()
            .all()
        )

    return [
        dict(row)
        for row in rows
    ]