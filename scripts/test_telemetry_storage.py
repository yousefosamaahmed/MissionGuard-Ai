from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from database.repositories.datasets import (
    create_dataset,
)
from database.repositories.missions import (
    create_mission,
)
from database.repositories.telemetry import (
    get_data_quality_report,
    list_feature_vectors,
    list_telemetry_samples,
    save_telemetry_batch,
)
from database.repositories.telemetry_sessions import (
    create_telemetry_session,
)


def main() -> None:
    print("=" * 70)
    print("MissionGuard Telemetry Storage Test")
    print("=" * 70)

    unique_suffix = (
        uuid4()
        .hex[:8]
        .upper()
    )

    print("\n1. Creating mission...")

    mission_id = create_mission(
        name="MissionGuard Telemetry Test",
        mission_code=(
            f"TEL-{unique_suffix}"
        ),
        spacecraft_name="ESA OPS-SAT",
        description=(
            "Telemetry samples and feature vectors test."
        ),
        status="active",
    )

    print(f"Mission ID: {mission_id}")

    print("\n2. Creating dataset...")

    dataset_id = create_dataset(
        name="Synthetic Telemetry Test Dataset",
        dataset_code=(
            f"SYN-{unique_suffix}"
        ),
        source_type="synthetic",
        source_organization="MissionGuard AI",
        description=(
            "Synthetic telemetry records for "
            "database integration testing."
        ),
        version="1.0",
        row_count=5,
        feature_count=6,
        is_labeled=True,
        metadata={
            "environment": "local",
            "purpose": "telemetry_storage_test",
        },
    )

    print(f"Dataset ID: {dataset_id}")

    print("\n3. Creating telemetry session...")

    telemetry_session_id = (
        create_telemetry_session(
            session_name=(
                "Synthetic Telemetry Storage Test"
            ),
            source_type="simulation",
            mission_id=mission_id,
            dataset_id=dataset_id,
            source_file_name=None,
            sampling_interval_seconds=60.0,
            total_samples=5,
            validation_status="valid",
            metadata={
                "test": True,
            },
        )
    )

    print(
        "Telemetry session ID: "
        f"{telemetry_session_id}"
    )

    base_timestamp = datetime(
        2026,
        1,
        1,
        12,
        0,
        0,
    )

    telemetry_records = [
        {
            "sample_index": 0,
            "timestamp": base_timestamp,
            "split_type": "upload",
            "ground_truth_label": False,
            "anomaly_type": None,
            "sample_metadata": {
                "source": "synthetic_test",
            },
            "feature_values": {
                "battery_temperature": 24.5,
                "battery_voltage": 28.1,
                "motor_temperature": 35.2,
                "pressure": 101.3,
                "vibration": 0.12,
                "radiation": 0.8,
            },
        },
        {
            "sample_index": 1,
            "timestamp": (
                base_timestamp
                + timedelta(minutes=1)
            ),
            "split_type": "upload",
            "ground_truth_label": False,
            "anomaly_type": None,
            "sample_metadata": {
                "source": "synthetic_test",
            },
            "feature_values": {
                "battery_temperature": 24.8,
                "battery_voltage": 28.0,
                "motor_temperature": 35.8,
                "pressure": 101.2,
                "vibration": 0.15,
                "radiation": 0.82,
            },
        },
        {
            "sample_index": 2,
            "timestamp": (
                base_timestamp
                + timedelta(minutes=2)
            ),
            "split_type": "upload",
            "ground_truth_label": False,
            "anomaly_type": None,
            "sample_metadata": {
                "source": "synthetic_test",
            },
            "feature_values": {
                "battery_temperature": 25.1,
                "battery_voltage": 27.9,
                "motor_temperature": 36.1,
                "pressure": 101.1,
                "vibration": 0.17,
                "radiation": 0.84,
            },
        },
        {
            "sample_index": 3,
            "timestamp": (
                base_timestamp
                + timedelta(minutes=3)
            ),
            "split_type": "upload",
            "ground_truth_label": True,
            "anomaly_type": (
                "thermal_voltage_anomaly"
            ),
            "sample_metadata": {
                "source": "synthetic_test",
                "severity": "high",
            },
            "feature_values": {
                "battery_temperature": 48.9,
                "battery_voltage": 21.4,
                "motor_temperature": 75.3,
                "pressure": 98.2,
                "vibration": 1.85,
                "radiation": 2.7,
            },
        },
        {
            "sample_index": 4,
            "timestamp": (
                base_timestamp
                + timedelta(minutes=4)
            ),
            "split_type": "upload",
            "ground_truth_label": False,
            "anomaly_type": None,
            "sample_metadata": {
                "source": "synthetic_test",
            },
            "feature_values": {
                "battery_temperature": 25.3,
                "battery_voltage": 27.8,
                "motor_temperature": 36.5,
                "pressure": 101.0,
                "vibration": 0.16,
                "radiation": 0.85,
            },
        },
    ]

    print("\n4. Saving telemetry batch...")

    save_result = save_telemetry_batch(
        session_id=telemetry_session_id,
        records=telemetry_records,
        feature_schema_name=(
            "missionguard_core"
        ),
        feature_schema_version="1.0",
        quality_report={
            "row_count": 5,
            "invalid_timestamps": 0,
            "duplicate_timestamps": 0,
            "long_missing_gaps": 0,
            "constant_sensors": [],
            "out_of_domain_values": {
                "battery_temperature": 1,
                "battery_voltage": 1,
                "motor_temperature": 1,
                "pressure": 1,
                "vibration": 1,
                "radiation": 1,
            },
            "missing_value_summary": {},
            "sampling_report": {
                "expected_interval_seconds": 60,
                "detected_interval_seconds": 60,
                "is_regular": True,
            },
            "validation_messages": [
                (
                    "One synthetic anomaly was "
                    "included intentionally."
                )
            ],
            "overall_status": "valid",
        },
    )

    print(
        f"Stored samples: "
        f"{save_result.inserted_samples}"
    )

    print(
        f"Stored feature vectors: "
        f"{save_result.inserted_feature_vectors}"
    )

    print(
        f"Quality report ID: "
        f"{save_result.quality_report_id}"
    )

    print("\n5. Reading stored samples...")

    stored_samples = list_telemetry_samples(
        telemetry_session_id
    )

    for sample in stored_samples:
        print(
            "-",
            f"index={sample['sample_index']}",
            f"id={sample['id']}",
            (
                f"anomaly="
                f"{sample['ground_truth_label']}"
            ),
        )

    print("\n6. Reading feature vectors...")

    stored_features = list_feature_vectors(
        telemetry_session_id
    )

    print(
        f"Feature vectors found: "
        f"{len(stored_features)}"
    )

    print("\n7. Reading quality report...")

    stored_report = get_data_quality_report(
        telemetry_session_id
    )

    if stored_report is None:
        raise RuntimeError(
            "Data-quality report was not found."
        )

    print(
        f"Quality status: "
        f"{stored_report['overall_status']}"
    )

    print(
        f"Quality row count: "
        f"{stored_report['row_count']}"
    )

    if len(stored_samples) != 5:
        raise RuntimeError(
            "Expected 5 telemetry samples."
        )

    if len(stored_features) != 5:
        raise RuntimeError(
            "Expected 5 feature vectors."
        )

    print(
        "\nTelemetry storage test "
        "completed successfully."
    )


if __name__ == "__main__":
    main()