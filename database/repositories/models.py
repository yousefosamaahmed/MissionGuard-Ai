from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import database_session
from database.tables import get_table

VALID_MODEL_TYPES = {
    "isolation_forest",
    "random_forest",
    "lstm_autoencoder",
    "supervised_ensemble",
    "forecasting",
    "hybrid",
    "other",
}

VALID_MODEL_STATUSES = {
    "training",
    "validated",
    "active",
    "archived",
    "failed",
}

VALID_ARTIFACT_TYPES = {
    "joblib",
    "pytorch",
    "onnx",
    "scaler",
    "config",
    "metadata",
    "feature_columns",
    "other",
}

VALID_STORAGE_PROVIDERS = {
    "local",
    "github",
    "ibm_cos",
    "other",
}


def _to_uuid(value: object) -> UUID:
    """
    Convert a PostgreSQL UUID value into a Python UUID.
    """

    if isinstance(value, UUID):
        return value

    return UUID(str(value))


def calculate_file_sha256(
    file_path: Path,
) -> str:
    """
    Calculate the SHA-256 hash of a file.
    """

    resolved_path = file_path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model artifact does not exist: {resolved_path}"
        )

    digest = hashlib.sha256()

    with resolved_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def create_model_version(
    model_name: str,
    model_type: str,
    version: str,
    training_dataset_id: UUID | None = None,
    description: str | None = None,
    status: str = "training",
    feature_schema: dict[str, Any] | None = None,
    training_parameters: dict[str, Any] | None = None,
    model_size_bytes: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> UUID:
    """
    Create or update a model version.

    The combination of model_name and version is unique.
    """

    clean_model_name = model_name.strip()
    clean_version = version.strip()

    if not clean_model_name:
        raise ValueError(
            "Model name cannot be empty."
        )

    if not clean_version:
        raise ValueError(
            "Model version cannot be empty."
        )

    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Invalid model type: {model_type}"
        )

    if status not in VALID_MODEL_STATUSES:
        raise ValueError(
            f"Invalid model status: {status}"
        )

    if (
        model_size_bytes is not None
        and model_size_bytes < 0
    ):
        raise ValueError(
            "Model size cannot be negative."
        )

    model_versions = get_table(
        "model_versions"
    )

    insert_statement = pg_insert(
        model_versions
    ).values(
        model_name=clean_model_name,
        model_type=model_type,
        version=clean_version,
        description=description,
        training_dataset_id=training_dataset_id,
        status=status,
        feature_schema=feature_schema or {},
        training_parameters=training_parameters or {},
        model_size_bytes=model_size_bytes,
        metadata=metadata or {},
    )

    statement = (
        insert_statement
        .on_conflict_do_update(
            constraint="uq_model_name_version",
            set_={
                "model_type": (
                    insert_statement.excluded.model_type
                ),
                "description": (
                    insert_statement.excluded.description
                ),
                "training_dataset_id": (
                    insert_statement
                    .excluded
                    .training_dataset_id
                ),
                "status": (
                    insert_statement.excluded.status
                ),
                "feature_schema": (
                    insert_statement
                    .excluded
                    .feature_schema
                ),
                "training_parameters": (
                    insert_statement
                    .excluded
                    .training_parameters
                ),
                "model_size_bytes": (
                    insert_statement
                    .excluded
                    .model_size_bytes
                ),
                "metadata": (
                    insert_statement.excluded.metadata
                ),
            },
        )
        .returning(
            model_versions.c.id
        )
    )

    with database_session() as session:
        model_version_id = session.execute(
            statement
        ).scalar_one()

    return _to_uuid(
        model_version_id
    )


def register_model_artifact(
    model_version_id: UUID,
    artifact_type: str,
    file_path: Path,
    storage_provider: str = "local",
    metadata: dict[str, Any] | None = None,
) -> UUID:
    """
    Register a model file in PostgreSQL.

    The physical model file remains on disk.
    PostgreSQL stores its path and metadata.
    """

    if artifact_type not in VALID_ARTIFACT_TYPES:
        raise ValueError(
            f"Invalid artifact type: {artifact_type}"
        )

    if storage_provider not in VALID_STORAGE_PROVIDERS:
        raise ValueError(
            "Invalid storage provider: "
            f"{storage_provider}"
        )

    resolved_path = (
        file_path
        .expanduser()
        .resolve()
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {resolved_path}"
        )

    model_artifacts = get_table(
        "model_artifacts"
    )

    file_path_value = str(
        resolved_path
    )

    insert_statement = pg_insert(
        model_artifacts
    ).values(
        model_version_id=model_version_id,
        artifact_type=artifact_type,
        file_path=file_path_value,
        storage_provider=storage_provider,
        file_size_bytes=(
            resolved_path.stat().st_size
        ),
        sha256_hash=calculate_file_sha256(
            resolved_path
        ),
        metadata=metadata or {},
    )

    statement = (
        insert_statement
        .on_conflict_do_update(
            constraint="uq_model_artifact",
            set_={
                "storage_provider": (
                    insert_statement
                    .excluded
                    .storage_provider
                ),
                "file_size_bytes": (
                    insert_statement
                    .excluded
                    .file_size_bytes
                ),
                "sha256_hash": (
                    insert_statement
                    .excluded
                    .sha256_hash
                ),
                "metadata": (
                    insert_statement
                    .excluded
                    .metadata
                ),
            },
        )
        .returning(
            model_artifacts.c.id
        )
    )

    with database_session() as session:
        artifact_id = session.execute(
            statement
        ).scalar_one()

    return _to_uuid(
        artifact_id
    )


def get_model_version(
    model_version_id: UUID,
) -> dict[str, Any] | None:
    """
    Return one model version by ID.
    """

    model_versions = get_table(
        "model_versions"
    )

    statement = (
        select(model_versions)
        .where(
            model_versions.c.id
            == model_version_id
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


def list_model_versions() -> list[dict[str, Any]]:
    """
    Return model versions from newest to oldest.
    """

    model_versions = get_table(
        "model_versions"
    )

    statement = (
        select(model_versions)
        .order_by(
            model_versions
            .c
            .created_at
            .desc()
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


def list_model_artifacts(
    model_version_id: UUID,
) -> list[dict[str, Any]]:
    """
    Return artifacts registered for a model version.
    """

    model_artifacts = get_table(
        "model_artifacts"
    )

    statement = (
        select(model_artifacts)
        .where(
            model_artifacts.c.model_version_id
            == model_version_id
        )
        .order_by(
            model_artifacts
            .c
            .created_at
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