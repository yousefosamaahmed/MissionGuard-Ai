from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import insert, select

from database.connection import database_session
from database.tables import get_table

VALID_DATASET_SOURCE_TYPES = {
    "upload",
    "opssat",
    "nasa_smap",
    "nasa_msl",
    "demo",
    "synthetic",
    "other",
}

VALID_FILE_ROLES = {
    "raw",
    "processed",
    "train",
    "validation",
    "test",
    "official_test",
    "labels",
    "predictions",
    "metadata",
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
    Convert a database UUID value to Python UUID.
    """

    if isinstance(value, UUID):
        return value

    return UUID(str(value))


def create_dataset(
    name: str,
    source_type: str,
    dataset_code: str | None = None,
    source_organization: str | None = None,
    source_url: str | None = None,
    license_name: str | None = None,
    description: str | None = None,
    version: str | None = None,
    row_count: int = 0,
    feature_count: int = 0,
    is_labeled: bool = False,
    metadata: dict[str, Any] | None = None,
) -> UUID:
    """
    Create a new dataset and return its generated UUID.
    """

    clean_name = name.strip()

    if not clean_name:
        raise ValueError(
            "Dataset name cannot be empty."
        )

    if source_type not in VALID_DATASET_SOURCE_TYPES:
        raise ValueError(
            f"Invalid dataset source type: {source_type}"
        )

    if row_count < 0:
        raise ValueError(
            "Dataset row_count cannot be negative."
        )

    if feature_count < 0:
        raise ValueError(
            "Dataset feature_count cannot be negative."
        )

    datasets_table = get_table(
        "datasets"
    )

    statement = (
        insert(datasets_table)
        .values(
            name=clean_name,
            dataset_code=(
                dataset_code.strip()
                if dataset_code
                else None
            ),
            source_type=source_type,
            source_organization=source_organization,
            source_url=source_url,
            license_name=license_name,
            description=description,
            version=version,
            row_count=row_count,
            feature_count=feature_count,
            is_labeled=is_labeled,
            metadata=metadata or {},
        )
        .returning(
            datasets_table.c.id
        )
    )

    with database_session() as session:
        dataset_id = session.execute(
            statement
        ).scalar_one()

    return _to_uuid(dataset_id)


def get_dataset(
    dataset_id: UUID,
) -> dict[str, Any] | None:
    """
    Return one dataset by ID.
    """

    datasets_table = get_table(
        "datasets"
    )

    statement = (
        select(datasets_table)
        .where(
            datasets_table.c.id
            == dataset_id
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


def list_datasets() -> list[dict[str, Any]]:
    """
    Return all datasets from newest to oldest.
    """

    datasets_table = get_table(
        "datasets"
    )

    statement = (
        select(datasets_table)
        .order_by(
            datasets_table
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


def create_dataset_file(
    dataset_id: UUID,
    file_name: str,
    file_role: str,
    file_path: str,
    storage_provider: str = "local",
    file_size_bytes: int | None = None,
    mime_type: str | None = None,
    sha256_hash: str | None = None,
    row_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> UUID:
    """
    Register a dataset file.

    The actual file remains on disk or object storage.
    PostgreSQL stores only its path and metadata.
    """

    clean_file_name = file_name.strip()
    clean_file_path = file_path.strip()

    if not clean_file_name:
        raise ValueError(
            "Dataset file name cannot be empty."
        )

    if not clean_file_path:
        raise ValueError(
            "Dataset file path cannot be empty."
        )

    if file_role not in VALID_FILE_ROLES:
        raise ValueError(
            f"Invalid dataset file role: {file_role}"
        )

    if storage_provider not in VALID_STORAGE_PROVIDERS:
        raise ValueError(
            "Invalid storage provider: "
            f"{storage_provider}"
        )

    if (
        file_size_bytes is not None
        and file_size_bytes < 0
    ):
        raise ValueError(
            "File size cannot be negative."
        )

    if (
        row_count is not None
        and row_count < 0
    ):
        raise ValueError(
            "File row count cannot be negative."
        )

    dataset_files_table = get_table(
        "dataset_files"
    )

    statement = (
        insert(dataset_files_table)
        .values(
            dataset_id=dataset_id,
            file_name=clean_file_name,
            file_role=file_role,
            file_path=clean_file_path,
            storage_provider=storage_provider,
            file_size_bytes=file_size_bytes,
            mime_type=mime_type,
            sha256_hash=sha256_hash,
            row_count=row_count,
            metadata=metadata or {},
        )
        .returning(
            dataset_files_table.c.id
        )
    )

    with database_session() as session:
        dataset_file_id = session.execute(
            statement
        ).scalar_one()

    return _to_uuid(dataset_file_id)


def list_dataset_files(
    dataset_id: UUID,
) -> list[dict[str, Any]]:
    """
    Return all files registered for a dataset.
    """

    dataset_files_table = get_table(
        "dataset_files"
    )

    statement = (
        select(dataset_files_table)
        .where(
            dataset_files_table.c.dataset_id
            == dataset_id
        )
        .order_by(
            dataset_files_table
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