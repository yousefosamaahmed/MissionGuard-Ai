from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from database.repositories.datasets import (
    list_datasets,
)
from database.repositories.model_metrics import (
    list_model_metrics,
    save_model_metric,
)
from database.repositories.models import (
    create_model_version,
)

METRICS_FILE = (
    PROJECT_ROOT
    / "models"
    / "opssat_metrics.csv"
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


REQUIRED_COLUMNS = {
    "Model",
    "Precision",
    "Recall",
    "F1",
    "MCC",
    "PR-AUC",
    "ROC-AUC",
    "False Alarms / 1000",
    "TN",
    "FP",
    "FN",
    "TP",
    "Threshold",
}


MODEL_CONFIGURATIONS: dict[
    str,
    dict[str, str],
] = {
    "OPSSAT Isolation Forest": {
        "model_type": "isolation_forest",
        "version": "1.0",
        "description": (
            "Unsupervised OPS-SAT segment-level "
            "anomaly-detection model."
        ),
    },
    "OPSSAT Supervised Random Forest": {
        "model_type": "random_forest",
        "version": "1.0",
        "description": (
            "Supervised OPS-SAT segment-level "
            "Random Forest anomaly detector."
        ),
    },
    "OPSSAT Hybrid": {
        "model_type": "hybrid",
        "version": "1.0",
        "description": (
            "Hybrid OPS-SAT anomaly detector "
            "combining supervised and "
            "unsupervised signals."
        ),
    },
}


def find_latest_opssat_dataset_id() -> UUID:
    """
    Return the newest registered OPS-SAT dataset.
    """

    datasets = list_datasets()

    for dataset in datasets:
        if dataset.get("source_type") != "opssat":
            continue

        dataset_id = dataset.get("id")

        if isinstance(dataset_id, UUID):
            return dataset_id

        return UUID(
            str(dataset_id)
        )

    raise RuntimeError(
        "No OPS-SAT dataset was found in PostgreSQL."
    )


def validate_metrics_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate OPS-SAT metrics before database import.
    """

    if dataframe.empty:
        raise ValueError(
            "The OPS-SAT metrics file is empty."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Metrics CSV is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if dataframe[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(
            "Metrics CSV contains missing values."
        )

    unsupported_models = (
        set(
            dataframe["Model"].astype(str)
        )
        - set(MODEL_CONFIGURATIONS)
    )

    if unsupported_models:
        raise ValueError(
            "Unsupported model names: "
            + ", ".join(
                sorted(unsupported_models)
            )
        )


def calculate_accuracy(
    true_negative: int,
    false_positive: int,
    false_negative: int,
    true_positive: int,
) -> float:
    """
    Calculate classification accuracy.
    """

    total = (
        true_negative
        + false_positive
        + false_negative
        + true_positive
    )

    if total == 0:
        raise ValueError(
            "Confusion matrix total cannot be zero."
        )

    return (
        true_negative
        + true_positive
    ) / total


def calculate_false_alarm_rate(
    true_negative: int,
    false_positive: int,
) -> float:
    """
    Calculate FP / (FP + TN).
    """

    normal_total = (
        true_negative
        + false_positive
    )

    if normal_total == 0:
        return 0.0

    return (
        false_positive
        / normal_total
    )


def calculate_specificity(
    true_negative: int,
    false_positive: int,
) -> float:
    """
    Calculate TN / (TN + FP).
    """

    normal_total = (
        true_negative
        + false_positive
    )

    if normal_total == 0:
        return 0.0

    return (
        true_negative
        / normal_total
    )


def import_metric_row(
    row: dict[str, Any],
    dataset_id: UUID,
) -> tuple[UUID, UUID]:
    """
    Register one model and its evaluation metrics.
    """

    model_name = str(
        row["Model"]
    ).strip()

    model_config = (
        MODEL_CONFIGURATIONS[
            model_name
        ]
    )

    true_negative = int(
        row["TN"]
    )

    false_positive = int(
        row["FP"]
    )

    false_negative = int(
        row["FN"]
    )

    true_positive = int(
        row["TP"]
    )

    total_evaluated = (
        true_negative
        + false_positive
        + false_negative
        + true_positive
    )

    model_version_id = create_model_version(
        model_name=model_name,
        model_type=model_config[
            "model_type"
        ],
        version=model_config[
            "version"
        ],
        training_dataset_id=dataset_id,
        description=model_config[
            "description"
        ],
        status="validated",
        feature_schema={
            "schema_name": (
                "opssat_segment_features"
            ),
            "schema_version": "1.0",
            "record_level": (
                "segment_features"
            ),
            "feature_count": len(
                FEATURE_COLUMNS
            ),
            "feature_columns": (
                FEATURE_COLUMNS
            ),
        },
        training_parameters={
            "source": (
                "existing_project_model"
            ),
            "evaluation_file": (
                METRICS_FILE.name
            ),
        },
        metadata={
            "dataset_family": "OPS-SAT",
            "evaluation_available": True,
        },
    )

    confusion_matrix = {
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
    }

    metric_id = save_model_metric(
        model_version_id=(
            model_version_id
        ),
        dataset_id=dataset_id,
        evaluation_type="other",
        split_name="validation",
        precision_score=float(
            row["Precision"]
        ),
        recall_score=float(
            row["Recall"]
        ),
        f1_score=float(
            row["F1"]
        ),
        accuracy_score=calculate_accuracy(
            true_negative=true_negative,
            false_positive=false_positive,
            false_negative=false_negative,
            true_positive=true_positive,
        ),
        mcc_score=float(
            row["MCC"]
        ),
        roc_auc=float(
            row["ROC-AUC"]
        ),
        pr_auc=float(
            row["PR-AUC"]
        ),
        false_alarm_rate=(
            calculate_false_alarm_rate(
                true_negative=true_negative,
                false_positive=false_positive,
            )
        ),
        false_alarms_per_1000=float(
            row["False Alarms / 1000"]
        ),
        confusion_matrix=confusion_matrix,
        extra_metrics={
            "threshold": float(
                row["Threshold"]
            ),
            "specificity": (
                calculate_specificity(
                    true_negative=true_negative,
                    false_positive=false_positive,
                )
            ),
            "total_evaluated": (
                total_evaluated
            ),
            "negative_samples": (
                true_negative
                + false_positive
            ),
            "positive_samples": (
                true_positive
                + false_negative
            ),
            "record_level": (
                "segment_features"
            ),
            "source_file": (
                METRICS_FILE.name
            ),
        },
    )

    return (
        model_version_id,
        metric_id,
    )


def main() -> None:
    print("=" * 76)
    print("MissionGuard OPS-SAT Metrics Import")
    print("=" * 76)

    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_FILE}"
        )

    print(
        f"\nMetrics file: {METRICS_FILE}"
    )

    dataframe = pd.read_csv(
        METRICS_FILE,
        low_memory=False,
    )

    validate_metrics_dataframe(
        dataframe
    )

    print(
        f"Models detected: {len(dataframe)}"
    )

    dataset_id = (
        find_latest_opssat_dataset_id()
    )

    print(
        f"OPS-SAT dataset ID: {dataset_id}"
    )

    print(
        "\nImporting model metrics..."
    )

    imported_model_ids: list[UUID] = []

    records = dataframe.to_dict(
        orient="records"
    )

    for row_number, row in enumerate(
        records,
        start=1,
    ):
        model_version_id, metric_id = (
            import_metric_row(
                row=row,
                dataset_id=dataset_id,
            )
        )

        imported_model_ids.append(
            model_version_id
        )

        print(
            f"\n{row_number}. {row['Model']}"
        )

        print(
            f"   Model version ID: "
            f"{model_version_id}"
        )

        print(
            f"   Metric ID: "
            f"{metric_id}"
        )

        print(
            f"   F1: "
            f"{float(row['F1']):.6f}"
        )

        print(
            f"   ROC-AUC: "
            f"{float(row['ROC-AUC']):.6f}"
        )

        print(
            "   False alarms / 1000: "
            f"{float(row['False Alarms / 1000']):.6f}"
        )

    stored_metrics = list_model_metrics(
        dataset_id=dataset_id
    )

    imported_metric_count = sum(
        1
        for metric in stored_metrics
        if metric.get("model_version_id")
        in imported_model_ids
        and metric.get("split_name")
        == "validation"
    )

    print(
        "\n" + "=" * 76
    )

    print(
        f"Imported models: "
        f"{len(imported_model_ids)}"
    )

    print(
        f"Matching stored metrics: "
        f"{imported_metric_count}"
    )

    if imported_metric_count < len(
        imported_model_ids
    ):
        raise RuntimeError(
            "Not all model metrics were stored."
        )

    print(
        "\nOPS-SAT metrics import "
        "completed successfully."
    )


if __name__ == "__main__":
    main()