from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from database.repositories.missions import (
    create_mission,
)
from database.services.opssat_import_service import (
    import_opssat_feature_csv,
)


def main() -> None:
    print("=" * 70)
    print("MissionGuard OPS-SAT Feature Import")
    print("=" * 70)

    unique_suffix = (
        uuid4().hex[:8].upper()
    )

    print(
        "\n1. Creating OPS-SAT mission..."
    )

    mission_id = create_mission(
        name="OPS-SAT Anomaly Detection Mission",
        mission_code=(
            f"OPS-IMPORT-{unique_suffix}"
        ),
        spacecraft_name="ESA OPS-SAT",
        description=(
            "Mission created for importing "
            "real OPS-SAT feature data."
        ),
        status="active",
    )

    print(
        f"Mission ID: {mission_id}"
    )

    csv_file_path = (
        PROJECT_ROOT
        / "data"
        / "opssat"
        / "processed"
        / "validation_features.csv"
    )

    print(
        "\n2. Importing OPS-SAT CSV..."
    )

    print(
        f"File: {csv_file_path}"
    )

    result = import_opssat_feature_csv(
        mission_id=mission_id,
        csv_file_path=csv_file_path,
    )

    print(
        "\n3. Import result"
    )

    print(
        f"Dataset ID: "
        f"{result.dataset_id}"
    )

    print(
        f"Dataset file ID: "
        f"{result.dataset_file_id}"
    )

    print(
        f"Telemetry session ID: "
        f"{result.telemetry_session_id}"
    )

    print(
        f"Quality report ID: "
        f"{result.quality_report_id}"
    )

    print(
        f"CSV rows: "
        f"{result.row_count}"
    )

    print(
        f"Feature count: "
        f"{result.feature_count}"
    )

    print(
        f"Stored samples: "
        f"{result.stored_samples}"
    )

    print(
        f"Stored feature vectors: "
        f"{result.stored_feature_vectors}"
    )

    if (
        result.stored_samples
        != result.row_count
    ):
        raise RuntimeError(
            "Stored sample count does not "
            "match CSV row count."
        )

    if (
        result.stored_feature_vectors
        != result.row_count
    ):
        raise RuntimeError(
            "Stored feature-vector count does "
            "not match CSV row count."
        )

    print(
        "\nOPS-SAT import completed successfully."
    )


if __name__ == "__main__":
    main()