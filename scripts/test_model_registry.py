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


from database.repositories.datasets import (
    list_datasets,
)
from database.repositories.models import (
    create_model_version,
    get_model_version,
    list_model_artifacts,
    list_model_versions,
    register_model_artifact,
)

FEATURE_COLUMNS = [
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
]


def find_latest_opssat_dataset_id():
    """
    Return the latest registered OPS-SAT dataset ID.
    """

    datasets = list_datasets()

    for dataset in datasets:
        if dataset.get("source_type") == "opssat":
            return dataset["id"]

    raise RuntimeError(
        "No OPS-SAT dataset was found in PostgreSQL."
    )


def find_model_artifact() -> Path | None:
    """
    Find an existing model artifact in the models folder.
    """

    models_directory = (
        PROJECT_ROOT
        / "models"
    )

    if not models_directory.exists():
        return None

    supported_extensions = {
        ".joblib",
        ".pkl",
        ".onnx",
        ".pt",
        ".pth",
    }

    for file_path in sorted(
        models_directory.rglob("*")
    ):
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in supported_extensions
        ):
            return file_path

    return None


def detect_artifact_type(
    file_path: Path,
) -> str:
    """
    Map a model file extension to the database type.
    """

    suffix = file_path.suffix.lower()

    if suffix in {
        ".joblib",
        ".pkl",
    }:
        return "joblib"

    if suffix == ".onnx":
        return "onnx"

    if suffix in {
        ".pt",
        ".pth",
    }:
        return "pytorch"

    return "other"


def main() -> None:
    print("=" * 70)
    print("MissionGuard Model Registry Test")
    print("=" * 70)

    print(
        "\n1. Finding latest OPS-SAT dataset..."
    )

    training_dataset_id = (
        find_latest_opssat_dataset_id()
    )

    print(
        f"Training dataset ID: "
        f"{training_dataset_id}"
    )

    model_artifact = (
        find_model_artifact()
    )

    model_size_bytes = (
        model_artifact.stat().st_size
        if model_artifact is not None
        else None
    )

    version = (
        "local-"
        + uuid4().hex[:8].lower()
    )

    print(
        "\n2. Registering model version..."
    )

    model_version_id = create_model_version(
        model_name=(
            "MissionGuard OPS-SAT "
            "Anomaly Detector"
        ),
        model_type="isolation_forest",
        version=version,
        description=(
            "OPS-SAT segment-level anomaly "
            "detection model."
        ),
        training_dataset_id=(
            training_dataset_id
        ),
        status="active",
        feature_schema={
            "schema_name": (
                "opssat_segment_features"
            ),
            "schema_version": "1.0",
            "feature_count": len(
                FEATURE_COLUMNS
            ),
            "feature_columns": (
                FEATURE_COLUMNS
            ),
        },
        training_parameters={
            "source": "existing_project_model",
            "record_level": (
                "segment_features"
            ),
        },
        model_size_bytes=(
            model_size_bytes
        ),
        metadata={
            "project": "MissionGuard AI",
            "dataset_family": "OPS-SAT",
            "environment": "local",
        },
    )

    print(
        f"Model version ID: "
        f"{model_version_id}"
    )

    if model_artifact is not None:
        print(
            "\n3. Registering model artifact..."
        )

        artifact_id = register_model_artifact(
            model_version_id=model_version_id,
            artifact_type=detect_artifact_type(
                model_artifact
            ),
            file_path=model_artifact,
            storage_provider="local",
            metadata={
                "original_file_name": (
                    model_artifact.name
                ),
            },
        )

        print(
            f"Artifact file: "
            f"{model_artifact}"
        )

        print(
            f"Artifact ID: "
            f"{artifact_id}"
        )

    else:
        print(
            "\n3. No .joblib, .pkl, .onnx, "
            ".pt or .pth model file was found."
        )

        print(
            "The model version was registered "
            "without an artifact."
        )

    print(
        "\n4. Reading stored model..."
    )

    stored_model = get_model_version(
        model_version_id
    )

    if stored_model is None:
        raise RuntimeError(
            "The registered model could not be found."
        )

    print(
        f"Model name: "
        f"{stored_model['model_name']}"
    )

    print(
        f"Model type: "
        f"{stored_model['model_type']}"
    )

    print(
        f"Version: "
        f"{stored_model['version']}"
    )

    print(
        f"Status: "
        f"{stored_model['status']}"
    )

    artifacts = list_model_artifacts(
        model_version_id
    )

    models = list_model_versions()

    print(
        f"Registered artifacts: "
        f"{len(artifacts)}"
    )

    print(
        f"Total model versions: "
        f"{len(models)}"
    )

    print(
        "\nModel registry test "
        "completed successfully."
    )


if __name__ == "__main__":
    main()