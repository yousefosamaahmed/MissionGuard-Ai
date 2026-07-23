from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from database.repositories.models import (
    list_model_versions,
    register_model_artifact,
)
from database.repositories.telemetry_sessions import (
    list_telemetry_sessions,
)
from database.services.opssat_inference_service import (
    run_real_opssat_analysis,
)

ARTIFACT_PATH = (
    PROJECT_ROOT
    / "models"
    / "opssat_model.joblib"
)


def to_uuid(
    value: object,
) -> UUID:
    """
    Convert a database UUID value to UUID.
    """

    if isinstance(value, UUID):
        return value

    return UUID(str(value))


def find_latest_opssat_session_id() -> UUID:
    """
    Return the latest OPS-SAT telemetry session.
    """

    sessions = list_telemetry_sessions()

    for session in sessions:
        if session.get("source_type") == "opssat":
            return to_uuid(
                session["id"]
            )

    raise RuntimeError(
        "No OPS-SAT telemetry session was found."
    )


def find_hybrid_model_version_id() -> UUID:
    """
    Return OPSSAT Hybrid model version 1.0.
    """

    models = list_model_versions()

    for model in models:
        if (
            model.get("model_name")
            == "OPSSAT Hybrid"
            and model.get("version") == "1.0"
        ):
            return to_uuid(
                model["id"]
            )

    raise RuntimeError(
        "OPSSAT Hybrid version 1.0 was not found. "
        "Run import_opssat_metrics.py first."
    )


def main() -> None:
    print("=" * 78)
    print("MissionGuard Real OPS-SAT Analysis")
    print("=" * 78)

    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Artifact not found: {ARTIFACT_PATH}"
        )

    print(
        "\n1. Finding OPS-SAT telemetry session..."
    )

    telemetry_session_id = (
        find_latest_opssat_session_id()
    )

    print(
        f"Telemetry session ID: "
        f"{telemetry_session_id}"
    )

    print(
        "\n2. Finding Hybrid model version..."
    )

    model_version_id = (
        find_hybrid_model_version_id()
    )

    print(
        f"Model version ID: "
        f"{model_version_id}"
    )

    print(
        "\n3. Registering model artifact..."
    )

    artifact_id = register_model_artifact(
        model_version_id=model_version_id,
        artifact_type="joblib",
        file_path=ARTIFACT_PATH,
        storage_provider="local",
        metadata={
            "artifact_role": (
                "complete_opssat_hybrid_bundle"
            ),
            "contains_isolation_model": True,
            "contains_supervised_model": True,
            "trusted_local_artifact": True,
        },
    )

    print(
        f"Artifact ID: {artifact_id}"
    )

    print(
        "\n4. Running real model inference..."
    )

    result = run_real_opssat_analysis(
        telemetry_session_id=(
            telemetry_session_id
        ),
        model_version_id=(
            model_version_id
        ),
        artifact_path=ARTIFACT_PATH,
    )

    print(
        "\n5. Analysis result"
    )

    print(
        f"Analysis run ID: "
        f"{result.analysis_run_id}"
    )

    print(
        f"Total predictions: "
        f"{result.total_predictions}"
    )

    print(
        f"Detected anomalies: "
        f"{result.total_anomalies}"
    )

    print(
        f"Created incidents: "
        f"{result.total_incidents}"
    )

    print(
        f"Mean risk score: "
        f"{result.mean_risk_score:.2f}"
    )

    print(
        f"Maximum risk score: "
        f"{result.maximum_risk_score:.2f}"
    )

    if result.total_predictions != 399:
        raise RuntimeError(
            "Expected 399 real predictions."
        )

    print(
        "\nReal OPS-SAT analysis "
        "completed successfully."
    )


if __name__ == "__main__":
    main()