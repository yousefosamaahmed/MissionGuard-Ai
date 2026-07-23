from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import insert, select

from database.connection import database_session
from database.tables import get_table

VALID_SESSION_SOURCE_TYPES = {
    "uploaded_csv",
    "demo",
    "opssat",
    "nasa",
    "replay",
    "simulation",
    "other",
}

VALID_VALIDATION_STATUSES = {
    "pending",
    "valid",
    "warning",
    "invalid",
}


def _to_uuid(value: object) -> UUID:
    """
    Convert a database UUID value to Python UUID.
    """

    if isinstance(value, UUID):
        return value

    return UUID(str(value))


def create_telemetry_session(
    session_name: str,
    source_type: str,
    mission_id: UUID | None = None,
    dataset_id: UUID | None = None,
    source_file_name: str | None = None,
    sampling_interval_seconds: float | None = None,
    total_samples: int = 0,
    validation_status: str = "pending",
    metadata: dict[str, Any] | None = None,
) -> UUID:
    """
    Create a telemetry ingestion or analysis session.
    """

    clean_session_name = session_name.strip()

    if not clean_session_name:
        raise ValueError(
            "Telemetry session name cannot be empty."
        )

    if mission_id is None and dataset_id is None:
        raise ValueError(
            "A telemetry session must reference "
            "a mission, a dataset, or both."
        )

    if source_type not in VALID_SESSION_SOURCE_TYPES:
        raise ValueError(
            f"Invalid session source type: {source_type}"
        )

    if validation_status not in VALID_VALIDATION_STATUSES:
        raise ValueError(
            "Invalid validation status: "
            f"{validation_status}"
        )

    if total_samples < 0:
        raise ValueError(
            "Total samples cannot be negative."
        )

    if (
        sampling_interval_seconds is not None
        and sampling_interval_seconds <= 0
    ):
        raise ValueError(
            "Sampling interval must be greater than zero."
        )

    sessions_table = get_table(
        "telemetry_sessions"
    )

    statement = (
        insert(sessions_table)
        .values(
            mission_id=mission_id,
            dataset_id=dataset_id,
            session_name=clean_session_name,
            source_type=source_type,
            source_file_name=source_file_name,
            sampling_interval_seconds=(
                sampling_interval_seconds
            ),
            total_samples=total_samples,
            validation_status=validation_status,
            metadata=metadata or {},
        )
        .returning(
            sessions_table.c.id
        )
    )

    with database_session() as session:
        telemetry_session_id = session.execute(
            statement
        ).scalar_one()

    return _to_uuid(telemetry_session_id)


def get_telemetry_session(
    telemetry_session_id: UUID,
) -> dict[str, Any] | None:
    """
    Return one telemetry session.
    """

    sessions_table = get_table(
        "telemetry_sessions"
    )

    statement = (
        select(sessions_table)
        .where(
            sessions_table.c.id
            == telemetry_session_id
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


def list_telemetry_sessions(
    mission_id: UUID | None = None,
    dataset_id: UUID | None = None,
) -> list[dict[str, Any]]:
    """
    Return telemetry sessions with optional filtering.
    """

    sessions_table = get_table(
        "telemetry_sessions"
    )

    statement = select(
        sessions_table
    )

    if mission_id is not None:
        statement = statement.where(
            sessions_table.c.mission_id
            == mission_id
        )

    if dataset_id is not None:
        statement = statement.where(
            sessions_table.c.dataset_id
            == dataset_id
        )

    statement = statement.order_by(
        sessions_table
        .c
        .created_at
        .desc()
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