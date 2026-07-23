from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from database.repositories.datasets import (
    create_dataset,
    create_dataset_file,
    get_dataset,
    list_dataset_files,
    list_datasets,
)
from database.repositories.missions import (
    create_mission,
)
from database.repositories.telemetry_sessions import (
    create_telemetry_session,
    get_telemetry_session,
    list_telemetry_sessions,
)


def count_csv_rows(
    file_path: Path,
) -> int:
    """
    Count CSV data rows without requiring pandas.
    """

    if not file_path.exists():
        return 0

    with file_path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        line_count = sum(
            1
            for _ in file
        )

    return max(
        line_count - 1,
        0,
    )


def main() -> None:
    print("=" * 70)
    print("MissionGuard Dataset Workflow Test")
    print("=" * 70)

    unique_suffix = (
        uuid4()
        .hex[:8]
        .upper()
    )

    print(
        "\n1. Creating mission..."
    )

    mission_id = create_mission(
        name=(
            "MissionGuard Dataset "
            "Integration Test"
        ),
        mission_code=(
            f"DATA-TEST-{unique_suffix}"
        ),
        spacecraft_name="ESA OPS-SAT",
        description=(
            "Dataset, file and telemetry "
            "session repository test."
        ),
        status="active",
    )

    print(
        f"Mission ID: {mission_id}"
    )

    demo_file_path = (
        PROJECT_ROOT
        / "data"
        / "demo"
        / "missionguard_demo_labeled.csv"
    )

    detected_row_count = count_csv_rows(
        demo_file_path
    )

    print(
        "\n2. Creating dataset..."
    )

    dataset_id = create_dataset(
        name=(
            "MissionGuard Competition "
            "Demo Dataset"
        ),
        dataset_code=(
            f"DEMO-{unique_suffix}"
        ),
        source_type="demo",
        source_organization=(
            "MissionGuard AI"
        ),
        license_name=(
            "Educational Prototype Dataset"
        ),
        description=(
            "Local demonstration telemetry "
            "dataset used for repository testing."
        ),
        version="1.0",
        row_count=detected_row_count,
        feature_count=6,
        is_labeled=True,
        metadata={
            "environment": "local",
            "purpose": "repository_test",
        },
    )

    print(
        f"Dataset ID: {dataset_id}"
    )

    print(
        "\n3. Registering dataset file..."
    )

    dataset_file_id = create_dataset_file(
        dataset_id=dataset_id,
        file_name=demo_file_path.name,
        file_role="processed",
        file_path=str(
            demo_file_path
        ),
        storage_provider="local",
        file_size_bytes=(
            demo_file_path.stat().st_size
            if demo_file_path.exists()
            else None
        ),
        mime_type="text/csv",
        row_count=detected_row_count,
        metadata={
            "exists_locally": (
                demo_file_path.exists()
            ),
        },
    )

    print(
        f"Dataset file ID: {dataset_file_id}"
    )

    print(
        "\n4. Creating telemetry session..."
    )

    telemetry_session_id = (
        create_telemetry_session(
            session_name=(
                "Competition Demo "
                "Telemetry Session"
            ),
            source_type="demo",
            mission_id=mission_id,
            dataset_id=dataset_id,
            source_file_name=(
                demo_file_path.name
            ),
            sampling_interval_seconds=60.0,
            total_samples=detected_row_count,
            validation_status="valid",
            metadata={
                "test_type": (
                    "dataset_workflow"
                ),
            },
        )
    )

    print(
        "Telemetry session ID: "
        f"{telemetry_session_id}"
    )

    print(
        "\n5. Reading the stored dataset..."
    )

    stored_dataset = get_dataset(
        dataset_id
    )

    if stored_dataset is None:
        raise RuntimeError(
            "The stored dataset could not be found."
        )

    print(
        f"Dataset name: "
        f"{stored_dataset['name']}"
    )

    print(
        f"Dataset rows: "
        f"{stored_dataset['row_count']}"
    )

    print(
        f"Dataset labeled: "
        f"{stored_dataset['is_labeled']}"
    )

    print(
        "\n6. Reading registered files..."
    )

    dataset_files = list_dataset_files(
        dataset_id
    )

    for dataset_file in dataset_files:
        print(
            "-",
            dataset_file["file_name"],
            "|",
            dataset_file["file_role"],
            "|",
            dataset_file["storage_provider"],
        )

    print(
        "\n7. Reading telemetry session..."
    )

    stored_session = get_telemetry_session(
        telemetry_session_id
    )

    if stored_session is None:
        raise RuntimeError(
            "The telemetry session could not be found."
        )

    print(
        f"Session name: "
        f"{stored_session['session_name']}"
    )

    print(
        f"Source type: "
        f"{stored_session['source_type']}"
    )

    print(
        f"Validation status: "
        f"{stored_session['validation_status']}"
    )

    mission_sessions = list_telemetry_sessions(
        mission_id=mission_id
    )

    dataset_sessions = list_telemetry_sessions(
        dataset_id=dataset_id
    )

    all_datasets = list_datasets()

    print(
        "\n8. Workflow summary"
    )

    print(
        f"Mission sessions: "
        f"{len(mission_sessions)}"
    )

    print(
        f"Dataset sessions: "
        f"{len(dataset_sessions)}"
    )

    print(
        f"Total stored datasets: "
        f"{len(all_datasets)}"
    )

    print(
        "\nDataset workflow completed successfully."
    )


if __name__ == "__main__":
    main()