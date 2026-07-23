from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import insert, select, update

from database.connection import database_session
from database.tables import get_table

VALID_EVALUATION_TYPES = {
    "timestamp_level",
    "event_level",
    "external_benchmark",
    "cross_validation",
    "other",
}


def _to_uuid(value: object) -> UUID:
    """
    Convert a PostgreSQL UUID value to Python UUID.
    """

    if isinstance(value, UUID):
        return value

    return UUID(str(value))


def _validate_probability(
    name: str,
    value: float | None,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> None:
    """
    Validate a score constrained to a numeric range.
    """

    if value is None:
        return

    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between "
            f"{minimum} and {maximum}. "
            f"Received: {value}"
        )


def save_model_metric(
    model_version_id: UUID,
    evaluation_type: str,
    dataset_id: UUID | None = None,
    split_name: str | None = None,
    precision_score: float | None = None,
    recall_score: float | None = None,
    f1_score: float | None = None,
    accuracy_score: float | None = None,
    mcc_score: float | None = None,
    roc_auc: float | None = None,
    pr_auc: float | None = None,
    false_alarm_rate: float | None = None,
    false_alarms_per_1000: float | None = None,
    mean_detection_delay: float | None = None,
    median_detection_delay: float | None = None,
    event_precision: float | None = None,
    event_recall: float | None = None,
    event_f1: float | None = None,
    detected_events: int | None = None,
    missed_events: int | None = None,
    false_event_alerts: int | None = None,
    duplicate_alerts: int | None = None,
    confusion_matrix: dict[str, Any] | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> UUID:
    """
    Insert or update one model-evaluation record.

    A record is treated as the same evaluation when its:

    - model_version_id
    - dataset_id
    - evaluation_type
    - split_name

    are identical.
    """

    if evaluation_type not in VALID_EVALUATION_TYPES:
        raise ValueError(
            "Invalid evaluation type: "
            f"{evaluation_type}"
        )

    score_values = {
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "accuracy_score": accuracy_score,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_alarm_rate": false_alarm_rate,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
    }

    for score_name, score_value in score_values.items():
        _validate_probability(
            score_name,
            score_value,
        )

    _validate_probability(
        "mcc_score",
        mcc_score,
        minimum=-1.0,
        maximum=1.0,
    )

    non_negative_values = {
        "false_alarms_per_1000": (
            false_alarms_per_1000
        ),
        "mean_detection_delay": (
            mean_detection_delay
        ),
        "median_detection_delay": (
            median_detection_delay
        ),
    }

    for value_name, numeric_value in (
        non_negative_values.items()
    ):
        if (
            numeric_value is not None
            and numeric_value < 0
        ):
            raise ValueError(
                f"{value_name} cannot be negative."
            )

    count_values = {
        "detected_events": detected_events,
        "missed_events": missed_events,
        "false_event_alerts": (
            false_event_alerts
        ),
        "duplicate_alerts": duplicate_alerts,
    }

    for count_name, count_value in count_values.items():
        if (
            count_value is not None
            and count_value < 0
        ):
            raise ValueError(
                f"{count_name} cannot be negative."
            )

    model_metrics = get_table(
        "model_metrics"
    )

    clean_split_name = (
        split_name.strip()
        if split_name
        else None
    )

    metric_values: dict[str, Any] = {
        "model_version_id": model_version_id,
        "dataset_id": dataset_id,
        "evaluation_type": evaluation_type,
        "split_name": clean_split_name,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "accuracy_score": accuracy_score,
        "mcc_score": mcc_score,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_alarm_rate": false_alarm_rate,
        "false_alarms_per_1000": (
            false_alarms_per_1000
        ),
        "mean_detection_delay": (
            mean_detection_delay
        ),
        "median_detection_delay": (
            median_detection_delay
        ),
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "detected_events": detected_events,
        "missed_events": missed_events,
        "false_event_alerts": (
            false_event_alerts
        ),
        "duplicate_alerts": duplicate_alerts,
        "confusion_matrix": (
            confusion_matrix or {}
        ),
        "extra_metrics": extra_metrics or {},
    }

    find_statement = (
        select(model_metrics.c.id)
        .where(
            model_metrics.c.model_version_id
            == model_version_id
        )
        .where(
            model_metrics.c.evaluation_type
            == evaluation_type
        )
    )

    if dataset_id is None:
        find_statement = find_statement.where(
            model_metrics.c.dataset_id.is_(None)
        )

    else:
        find_statement = find_statement.where(
            model_metrics.c.dataset_id
            == dataset_id
        )

    if clean_split_name is None:
        find_statement = find_statement.where(
            model_metrics.c.split_name.is_(None)
        )

    else:
        find_statement = find_statement.where(
            model_metrics.c.split_name
            == clean_split_name
        )

    with database_session() as session:
        existing_metric_id = (
            session.execute(
                find_statement
            )
            .scalar_one_or_none()
        )

        if existing_metric_id is None:
            statement = (
                insert(model_metrics)
                .values(**metric_values)
                .returning(
                    model_metrics.c.id
                )
            )

        else:
            statement = (
                update(model_metrics)
                .where(
                    model_metrics.c.id
                    == existing_metric_id
                )
                .values(**metric_values)
                .returning(
                    model_metrics.c.id
                )
            )

        metric_id = session.execute(
            statement
        ).scalar_one()

    return _to_uuid(
        metric_id
    )


def list_model_metrics(
    model_version_id: UUID | None = None,
    dataset_id: UUID | None = None,
) -> list[dict[str, Any]]:
    """
    Return stored model metrics with optional filters.
    """

    model_metrics = get_table(
        "model_metrics"
    )

    statement = select(
        model_metrics
    )

    if model_version_id is not None:
        statement = statement.where(
            model_metrics.c.model_version_id
            == model_version_id
        )

    if dataset_id is not None:
        statement = statement.where(
            model_metrics.c.dataset_id
            == dataset_id
        )

    statement = statement.order_by(
        model_metrics
        .c
        .evaluated_at
        .desc()
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