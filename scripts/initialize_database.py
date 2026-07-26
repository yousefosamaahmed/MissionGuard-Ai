from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.repositories.datasets import list_datasets
from database.repositories.missions import create_mission, list_missions
from database.repositories.models import register_model_artifact
from database.repositories.telemetry_sessions import list_telemetry_sessions
from database.services.opssat_import_service import import_opssat_feature_csv
from scripts.import_opssat_metrics import (
    METRICS_FILE,
    import_metric_row,
    validate_metrics_dataframe,
)

MISSION_CODE = "OPS-SAT-OFFICIAL"
DATASET_CODE = "OPSSAT-AD-VALIDATION-V1"


def _to_uuid(value: object) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _find_mission_id() -> UUID | None:
    for mission in list_missions():
        if mission.get("mission_code") == MISSION_CODE:
            return _to_uuid(mission["id"])
    return None


def _find_dataset() -> dict[str, Any] | None:
    for dataset in list_datasets():
        if dataset.get("dataset_code") == DATASET_CODE:
            return dataset
    return None


def _ensure_official_dataset() -> tuple[UUID, UUID]:
    mission_id = _find_mission_id()

    if mission_id is None:
        mission_id = create_mission(
            name="OPS-SAT Anomaly Detection Mission",
            mission_code=MISSION_CODE,
            spacecraft_name="ESA OPS-SAT",
            description=(
                "Official MissionGuard mission record for the OPSSAT-AD "
                "benchmark and persisted analysis runs."
            ),
            status="active",
        )
        print(f"Created official mission: {mission_id}")
    else:
        print(f"Official mission already exists: {mission_id}")

    dataset = _find_dataset()

    if dataset is None:
        csv_file_path = (
            PROJECT_ROOT
            / "data"
            / "opssat"
            / "processed"
            / "validation_features.csv"
        )
        result = import_opssat_feature_csv(
            mission_id=mission_id,
            csv_file_path=csv_file_path,
            dataset_name="OPSSAT-AD Validation Features",
            dataset_code=DATASET_CODE,
        )
        print(
            "Imported official OPS-SAT dataset: "
            f"{result.row_count} rows, session {result.telemetry_session_id}"
        )
        return result.dataset_id, result.telemetry_session_id

    dataset_id = _to_uuid(dataset["id"])
    sessions = list_telemetry_sessions(dataset_id=dataset_id)

    official_sessions = [
        session
        for session in sessions
        if session.get("source_type") == "opssat"
    ]

    if not official_sessions:
        raise RuntimeError(
            "The official dataset record exists but its telemetry session is "
            "missing. Remove the partial OPSSAT-AD-VALIDATION-V1 dataset in "
            "pgAdmin and restart the application to rebuild it safely."
        )

    session_id = _to_uuid(official_sessions[0]["id"])
    print(f"Official OPS-SAT dataset already exists: {dataset_id}")
    return dataset_id, session_id


def _register_metrics_and_artifacts(dataset_id: UUID) -> None:
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}")

    dataframe = pd.read_csv(METRICS_FILE, low_memory=False)
    validate_metrics_dataframe(dataframe)

    artifact_map = {
        "OPSSAT Isolation Forest": (
            PROJECT_ROOT / "models" / "opssat_isolation_bundle.joblib"
        ),
        "OPSSAT Supervised Random Forest": (
            PROJECT_ROOT / "models" / "opssat_supervised_bundle.joblib"
        ),
        "OPSSAT Hybrid": PROJECT_ROOT / "models" / "opssat_model.joblib",
    }

    for row in dataframe.to_dict(orient="records"):
        model_version_id, metric_id = import_metric_row(
            row=row,
            dataset_id=dataset_id,
        )
        model_name = str(row["Model"])
        artifact_path = artifact_map[model_name]

        if artifact_path.exists():
            register_model_artifact(
                model_version_id=model_version_id,
                artifact_type="joblib",
                file_path=artifact_path,
                storage_provider="local",
                metadata={
                    "model_name": model_name,
                    "server_initialization": True,
                },
            )

        print(
            f"Registered {model_name}: model={model_version_id}, "
            f"metric={metric_id}"
        )


def main() -> None:
    auto_seed = os.getenv("DATABASE_AUTO_SEED", "true").strip().lower()

    if auto_seed not in {"1", "true", "yes", "on"}:
        print("DATABASE_AUTO_SEED is disabled; skipping reference-data seed.")
        return

    dataset_id, session_id = _ensure_official_dataset()
    _register_metrics_and_artifacts(dataset_id)

    print(
        "MissionGuard database initialization is complete. "
        f"Dataset={dataset_id}, telemetry_session={session_id}."
    )


if __name__ == "__main__":
    main()
