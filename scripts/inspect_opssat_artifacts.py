from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.opssat import (
    load_artifact,
    predict_feature_rows,
)

MODELS_DIRECTORY = (
    PROJECT_ROOT
    / "models"
)

FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "opssat"
    / "processed"
    / "validation_features.csv"
)

ARTIFACT_PATHS = (
    MODELS_DIRECTORY
    / "opssat_model.joblib",

    MODELS_DIRECTORY
    / "opssat_isolation_bundle.joblib",

    MODELS_DIRECTORY
    / "opssat_supervised_bundle.joblib",
)


def print_dataframe(
    dataframe: pd.DataFrame,
    label: str,
) -> None:
    """
    Print a compact DataFrame summary.
    """

    print(f"\n{label}:")

    print(
        f"- Rows: {len(dataframe)}"
    )

    print(
        f"- Columns: "
        f"{list(dataframe.columns)}"
    )

    if not dataframe.empty:
        print("\nPreview:")

        with pd.option_context(
            "display.max_columns",
            None,
            "display.width",
            220,
        ):
            print(
                dataframe
                .head(5)
                .to_string(index=False)
            )


def print_result(
    result: Any,
    label: str = "Prediction result",
) -> None:
    """
    Print an inference result without assuming its type.
    """

    print(
        f"\n{label} type: "
        f"{type(result).__name__}"
    )

    if isinstance(
        result,
        pd.DataFrame,
    ):
        print_dataframe(
            result,
            label,
        )

        return

    if isinstance(
        result,
        pd.Series,
    ):
        print(
            f"- Name: {result.name!r}"
        )

        print(
            f"- Length: {len(result)}"
        )

        print(
            result.head(5).to_string()
        )

        return

    if isinstance(
        result,
        tuple,
    ):
        print(
            f"- Tuple items: {len(result)}"
        )

        for index, item in enumerate(
            result,
            start=1,
        ):
            print_result(
                item,
                label=f"Tuple item {index}",
            )

        return

    if isinstance(
        result,
        dict,
    ):
        print(
            f"- Keys: {list(result.keys())}"
        )

        for key, value in result.items():
            print(
                f"- {key!r}: "
                f"type={type(value).__name__}"
            )

        return

    print(
        f"- Value: {result!r}"
    )


def inspect_artifact(
    artifact_path: Path,
    feature_frame: pd.DataFrame,
) -> None:
    """
    Inspect one OPS-SAT model artifact.
    """

    print("\n" + "=" * 90)
    print(
        f"ARTIFACT: {artifact_path.name}"
    )
    print("=" * 90)

    if not artifact_path.exists():
        print(
            f"ERROR: File does not exist: "
            f"{artifact_path}"
        )

        return

    print(
        f"Path: {artifact_path}"
    )

    print(
        f"Size: "
        f"{artifact_path.stat().st_size} bytes"
    )

    try:
        artifact = load_artifact(
            artifact_path
        )

    except Exception as error:
        print(
            "This file was rejected by "
            "load_artifact()."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error message: "
            f"{error}"
        )

        return

    print(
        "\nArtifact accepted by load_artifact()."
    )

    print(
        f"Keys: "
        f"{sorted(artifact.keys())}"
    )

    numeric_features = artifact.get(
        "numeric_features",
        [],
    )

    thresholds = artifact.get(
        "thresholds",
        {},
    )

    channels = artifact.get(
        "channels",
        [],
    )

    print(
        "\nCore artifact information:"
    )

    print(
        f"- Isolation model type: "
        f"{type(artifact.get('isolation_model')).__name__}"
    )

    print(
        f"- Supervised model type: "
        f"{type(artifact.get('supervised_model')).__name__}"
    )

    print(
        f"- Numeric features: "
        f"{list(numeric_features)}"
    )

    print(
        f"- Numeric feature count: "
        f"{len(numeric_features)}"
    )

    print(
        f"- Thresholds: "
        f"{thresholds!r}"
    )

    print(
        f"- Trained channel count: "
        f"{len(channels)}"
    )

    missing_features = [
        feature_name
        for feature_name in numeric_features
        if feature_name not in feature_frame.columns
    ]

    if missing_features:
        print(
            "\nERROR: Validation dataset is missing "
            "model features:"
        )

        for feature_name in missing_features:
            print(
                f"- {feature_name}"
            )

        return

    sample_frame = (
        feature_frame
        .head(5)
        .copy()
    )

    print(
        "\nRunning predict_feature_rows() "
        "on the first five rows..."
    )

    try:
        prediction_result = (
            predict_feature_rows(
                sample_frame,
                artifact,
            )
        )

    except Exception as error:
        print(
            "Inference failed."
        )

        print(
            f"Error type: "
            f"{type(error).__name__}"
        )

        print(
            f"Error message: "
            f"{error}"
        )

        return

    print_result(
        prediction_result
    )


def main() -> None:
    print("=" * 90)
    print("MissionGuard OPS-SAT Artifact Inspection")
    print("=" * 90)

    print(
        "\npredict_feature_rows signature:"
    )

    print(
        inspect.signature(
            predict_feature_rows
        )
    )

    print(
        "\nload_artifact signature:"
    )

    print(
        inspect.signature(
            load_artifact
        )
    )

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            "OPS-SAT validation feature file "
            f"was not found: {FEATURES_FILE}"
        )

    feature_frame = pd.read_csv(
        FEATURES_FILE,
        low_memory=False,
    )

    print_dataframe(
        feature_frame.head(5),
        "Validation input sample",
    )

    for artifact_path in ARTIFACT_PATHS:
        inspect_artifact(
            artifact_path,
            feature_frame,
        )

    print("\n" + "=" * 90)
    print("Artifact inspection completed.")
    print("=" * 90)


if __name__ == "__main__":
    main()