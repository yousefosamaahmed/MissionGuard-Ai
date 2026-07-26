from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import case, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.connection import database_session
from database.tables import get_table

VALID_RUN_TYPES = {
    "live_analysis",
    "benchmark",
    "replay",
    "simulation",
    "championship_demo",
    "other",
}

VALID_RUN_STATUSES = {
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
}

VALID_RISK_LEVELS = {
    "Normal",
    "Watch",
    "Warning",
    "Critical",
}

VALID_INCIDENT_SEVERITIES = {
    "Watch",
    "Warning",
    "Critical",
}

VALID_INCIDENT_STATUSES = {
    "open",
    "under_review",
    "confirmed",
    "resolved",
    "rejected",
}


@dataclass(frozen=True)
class PredictionSaveResult:
    """
    Summary returned after storing predictions.
    """

    stored_predictions: int
    anomaly_predictions: int
    prediction_ids_by_sample: dict[int, int]


def _to_uuid(value: object) -> UUID:
    """
    Convert a PostgreSQL UUID value into Python UUID.
    """

    if isinstance(value, UUID):
        return value

    return UUID(str(value))


def _validate_percentage(
    name: str,
    value: float | None,
) -> None:
    """
    Validate a score between 0 and 100.
    """

    if value is None:
        return

    if not 0.0 <= value <= 100.0:
        raise ValueError(
            f"{name} must be between 0 and 100. "
            f"Received: {value}"
        )


def create_analysis_run(
    session_id: UUID,
    model_version_id: UUID,
    run_type: str,
    status: str = "running",
    metadata: dict[str, Any] | None = None,
) -> UUID:
    """
    Create a new model-analysis run.
    """

    if run_type not in VALID_RUN_TYPES:
        raise ValueError(
            f"Invalid analysis run type: {run_type}"
        )

    if status not in VALID_RUN_STATUSES:
        raise ValueError(
            f"Invalid analysis run status: {status}"
        )

    analysis_runs = get_table(
        "analysis_runs"
    )

    statement = (
        pg_insert(analysis_runs)
        .values(
            session_id=session_id,
            model_version_id=model_version_id,
            run_type=run_type,
            status=status,
            total_predictions=0,
            total_anomalies=0,
            total_incidents=0,
            metadata=metadata or {},
        )
        .returning(
            analysis_runs.c.id
        )
    )

    with database_session() as session:
        analysis_run_id = session.execute(
            statement
        ).scalar_one()

    return _to_uuid(
        analysis_run_id
    )


def save_prediction_batch(
    analysis_run_id: UUID,
    prediction_records: list[dict[str, Any]],
) -> PredictionSaveResult:
    """
    Store multiple predictions using a bulk upsert.

    The unique constraint prevents duplicate predictions
    for the same analysis run and telemetry sample.
    """

    if not prediction_records:
        raise ValueError(
            "Prediction records cannot be empty."
        )

    predictions = get_table(
        "predictions"
    )

    prediction_rows: list[dict[str, Any]] = []

    seen_sample_ids: set[int] = set()

    for record in prediction_records:
        telemetry_sample_id = int(
            record["telemetry_sample_id"]
        )

        if telemetry_sample_id in seen_sample_ids:
            raise ValueError(
                "Duplicate telemetry_sample_id inside "
                f"prediction batch: {telemetry_sample_id}"
            )

        seen_sample_ids.add(
            telemetry_sample_id
        )

        risk_level = str(
            record["risk_level"]
        )

        if risk_level not in VALID_RISK_LEVELS:
            raise ValueError(
                f"Invalid risk level: {risk_level}"
            )

        risk_score = float(
            record["risk_score"]
        )

        confidence_score = float(
            record["confidence_score"]
        )

        _validate_percentage(
            "risk_score",
            risk_score,
        )

        _validate_percentage(
            "confidence_score",
            confidence_score,
        )

        optional_scores = {
            "isolation_score": record.get(
                "isolation_score"
            ),
            "forecast_residual_score": record.get(
                "forecast_residual_score"
            ),
            "rule_score": record.get(
                "rule_score"
            ),
            "persistence_score": record.get(
                "persistence_score"
            ),
            "early_warning_score": record.get(
                "early_warning_score"
            ),
        }

        for score_name, score_value in (
            optional_scores.items()
        ):
            if score_value is not None:
                optional_scores[score_name] = float(
                    score_value
                )

                _validate_percentage(
                    score_name,
                    float(score_value),
                )

        affected_subsystems = record.get(
            "affected_subsystems",
            [],
        )

        rule_violations = record.get(
            "rule_violations",
            [],
        )

        if not isinstance(
            affected_subsystems,
            list,
        ):
            raise TypeError(
                "affected_subsystems must be a list."
            )

        if not isinstance(
            rule_violations,
            list,
        ):
            raise TypeError(
                "rule_violations must be a list."
            )

        prediction_rows.append(
            {
                "analysis_run_id": analysis_run_id,
                "telemetry_sample_id": (
                    telemetry_sample_id
                ),
                "predicted_anomaly": bool(
                    record["predicted_anomaly"]
                ),
                "risk_level": risk_level,
                "risk_score": risk_score,
                "confidence_score": (
                    confidence_score
                ),
                "isolation_score": (
                    optional_scores[
                        "isolation_score"
                    ]
                ),
                "forecast_residual_score": (
                    optional_scores[
                        "forecast_residual_score"
                    ]
                ),
                "rule_score": (
                    optional_scores[
                        "rule_score"
                    ]
                ),
                "persistence_score": (
                    optional_scores[
                        "persistence_score"
                    ]
                ),
                "early_warning_score": (
                    optional_scores[
                        "early_warning_score"
                    ]
                ),
                "top_feature": record.get(
                    "top_feature"
                ),
                "out_of_distribution": bool(
                    record.get(
                        "out_of_distribution",
                        False,
                    )
                ),
                "human_review_required": bool(
                    record.get(
                        "human_review_required",
                        False,
                    )
                ),
                "explanation": record.get(
                    "explanation"
                ),
                "feature_contributions": (
                    record.get(
                        "feature_contributions",
                        {},
                    )
                ),
                "rule_violations": (
                    rule_violations
                ),
                "affected_subsystems": (
                    affected_subsystems
                ),
                "prediction_metadata": (
                    record.get(
                        "prediction_metadata",
                        {},
                    )
                ),
            }
        )

    insert_statement = pg_insert(
        predictions
    ).values(
        prediction_rows
    )

    statement = (
        insert_statement
        .on_conflict_do_update(
            constraint=(
                "uq_analysis_sample_prediction"
            ),
            set_={
                "predicted_anomaly": (
                    insert_statement
                    .excluded
                    .predicted_anomaly
                ),
                "risk_level": (
                    insert_statement
                    .excluded
                    .risk_level
                ),
                "risk_score": (
                    insert_statement
                    .excluded
                    .risk_score
                ),
                "confidence_score": (
                    insert_statement
                    .excluded
                    .confidence_score
                ),
                "isolation_score": (
                    insert_statement
                    .excluded
                    .isolation_score
                ),
                "forecast_residual_score": (
                    insert_statement
                    .excluded
                    .forecast_residual_score
                ),
                "rule_score": (
                    insert_statement
                    .excluded
                    .rule_score
                ),
                "persistence_score": (
                    insert_statement
                    .excluded
                    .persistence_score
                ),
                "early_warning_score": (
                    insert_statement
                    .excluded
                    .early_warning_score
                ),
                "top_feature": (
                    insert_statement
                    .excluded
                    .top_feature
                ),
                "out_of_distribution": (
                    insert_statement
                    .excluded
                    .out_of_distribution
                ),
                "human_review_required": (
                    insert_statement
                    .excluded
                    .human_review_required
                ),
                "explanation": (
                    insert_statement
                    .excluded
                    .explanation
                ),
                "feature_contributions": (
                    insert_statement
                    .excluded
                    .feature_contributions
                ),
                "rule_violations": (
                    insert_statement
                    .excluded
                    .rule_violations
                ),
                "affected_subsystems": (
                    insert_statement
                    .excluded
                    .affected_subsystems
                ),
                "prediction_metadata": (
                    insert_statement
                    .excluded
                    .prediction_metadata
                ),
            },
        )
        .returning(
            predictions.c.id,
            predictions.c.telemetry_sample_id,
            predictions.c.predicted_anomaly,
        )
    )

    with database_session() as session:
        stored_rows = (
            session.execute(statement)
            .mappings()
            .all()
        )

    if len(stored_rows) != len(
        prediction_records
    ):
        raise RuntimeError(
            "Stored prediction count does not match "
            "the submitted prediction count."
        )

    prediction_ids_by_sample: dict[int, int] = {}

    anomaly_predictions = 0

    for stored_row in stored_rows:
        telemetry_sample_id = int(
            stored_row["telemetry_sample_id"]
        )

        prediction_id = int(
            stored_row["id"]
        )

        prediction_ids_by_sample[
            telemetry_sample_id
        ] = prediction_id

        if bool(
            stored_row["predicted_anomaly"]
        ):
            anomaly_predictions += 1

    return PredictionSaveResult(
        stored_predictions=len(stored_rows),
        anomaly_predictions=anomaly_predictions,
        prediction_ids_by_sample=(
            prediction_ids_by_sample
        ),
    )


def create_incident(
    analysis_run_id: UUID,
    incident_code: str,
    severity: str,
    peak_risk_score: float,
    start_sample_id: int | None = None,
    end_sample_id: int | None = None,
    started_at: Any | None = None,
    ended_at: Any | None = None,
    duration_samples: int | None = None,
    peak_confidence: float | None = None,
    top_feature: str | None = None,
    affected_subsystems: list[str] | None = None,
    human_review_required: bool = False,
    status: str = "open",
    summary: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> UUID:
    """
    Create or update an incident belonging to an analysis run.
    """

    clean_incident_code = incident_code.strip()

    if not clean_incident_code:
        raise ValueError(
            "Incident code cannot be empty."
        )

    if severity not in VALID_INCIDENT_SEVERITIES:
        raise ValueError(
            f"Invalid incident severity: {severity}"
        )

    if status not in VALID_INCIDENT_STATUSES:
        raise ValueError(
            f"Invalid incident status: {status}"
        )

    _validate_percentage(
        "peak_risk_score",
        peak_risk_score,
    )

    _validate_percentage(
        "peak_confidence",
        peak_confidence,
    )

    if (
        duration_samples is not None
        and duration_samples < 1
    ):
        raise ValueError(
            "duration_samples must be at least 1."
        )

    incidents = get_table(
        "incidents"
    )

    insert_statement = pg_insert(
        incidents
    ).values(
        analysis_run_id=analysis_run_id,
        incident_code=clean_incident_code,
        start_sample_id=start_sample_id,
        end_sample_id=end_sample_id,
        started_at=started_at,
        ended_at=ended_at,
        duration_samples=duration_samples,
        severity=severity,
        peak_risk_score=peak_risk_score,
        peak_confidence=peak_confidence,
        top_feature=top_feature,
        affected_subsystems=(
            affected_subsystems or []
        ),
        human_review_required=(
            human_review_required
        ),
        status=status,
        summary=summary,
        metadata=metadata or {},
    )

    statement = (
        insert_statement
        .on_conflict_do_update(
            constraint="uq_incident_code",
            set_={
                "start_sample_id": (
                    insert_statement
                    .excluded
                    .start_sample_id
                ),
                "end_sample_id": (
                    insert_statement
                    .excluded
                    .end_sample_id
                ),
                "started_at": (
                    insert_statement
                    .excluded
                    .started_at
                ),
                "ended_at": (
                    insert_statement
                    .excluded
                    .ended_at
                ),
                "duration_samples": (
                    insert_statement
                    .excluded
                    .duration_samples
                ),
                "severity": (
                    insert_statement
                    .excluded
                    .severity
                ),
                "peak_risk_score": (
                    insert_statement
                    .excluded
                    .peak_risk_score
                ),
                "peak_confidence": (
                    insert_statement
                    .excluded
                    .peak_confidence
                ),
                "top_feature": (
                    insert_statement
                    .excluded
                    .top_feature
                ),
                "affected_subsystems": (
                    insert_statement
                    .excluded
                    .affected_subsystems
                ),
                "human_review_required": (
                    insert_statement
                    .excluded
                    .human_review_required
                ),
                "status": (
                    insert_statement
                    .excluded
                    .status
                ),
                "summary": (
                    insert_statement
                    .excluded
                    .summary
                ),
                "metadata": (
                    insert_statement
                    .excluded
                    .metadata
                ),
                "updated_at": func.now(),
            },
        )
        .returning(
            incidents.c.id
        )
    )

    with database_session() as session:
        incident_id = session.execute(
            statement
        ).scalar_one()

    return _to_uuid(
        incident_id
    )


def complete_analysis_run(
    analysis_run_id: UUID,
    total_predictions: int,
    total_anomalies: int,
    total_incidents: int,
    mission_health_score: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Mark an analysis run as completed.
    """

    if total_predictions < 0:
        raise ValueError(
            "total_predictions cannot be negative."
        )

    if total_anomalies < 0:
        raise ValueError(
            "total_anomalies cannot be negative."
        )

    if total_incidents < 0:
        raise ValueError(
            "total_incidents cannot be negative."
        )

    if total_anomalies > total_predictions:
        raise ValueError(
            "total_anomalies cannot exceed "
            "total_predictions."
        )

    _validate_percentage(
        "mission_health_score",
        mission_health_score,
    )

    analysis_runs = get_table(
        "analysis_runs"
    )

    values: dict[str, Any] = {
        "status": "completed",
        "total_predictions": total_predictions,
        "total_anomalies": total_anomalies,
        "total_incidents": total_incidents,
        "mission_health_score": (
            mission_health_score
        ),
        "finished_at": func.now(),
        "error_message": None,
    }

    if metadata is not None:
        values["metadata"] = metadata

    statement = (
        update(analysis_runs)
        .where(
            analysis_runs.c.id
            == analysis_run_id
        )
        .values(**values)
        .returning(
            analysis_runs.c.id
        )
    )

    with database_session() as session:
        updated_id = session.execute(
            statement
        ).scalar_one_or_none()

    if updated_id is None:
        raise RuntimeError(
            f"Analysis run was not found: "
            f"{analysis_run_id}"
        )


def fail_analysis_run(
    analysis_run_id: UUID,
    error_message: str,
) -> None:
    """
    Mark an analysis run as failed.
    """

    clean_error_message = error_message.strip()

    if not clean_error_message:
        raise ValueError(
            "Error message cannot be empty."
        )

    analysis_runs = get_table(
        "analysis_runs"
    )

    statement = (
        update(analysis_runs)
        .where(
            analysis_runs.c.id
            == analysis_run_id
        )
        .values(
            status="failed",
            finished_at=func.now(),
            error_message=clean_error_message,
        )
        .returning(
            analysis_runs.c.id
        )
    )

    with database_session() as session:
        updated_id = session.execute(
            statement
        ).scalar_one_or_none()

    if updated_id is None:
        raise RuntimeError(
            f"Analysis run was not found: "
            f"{analysis_run_id}"
        )


def get_analysis_run(
    analysis_run_id: UUID,
) -> dict[str, Any] | None:
    """
    Return one analysis run.
    """

    analysis_runs = get_table(
        "analysis_runs"
    )

    statement = (
        select(analysis_runs)
        .where(
            analysis_runs.c.id
            == analysis_run_id
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


def list_predictions(
    analysis_run_id: UUID,
) -> list[dict[str, Any]]:
    """
    Return predictions for one analysis run.
    """

    predictions = get_table(
        "predictions"
    )

    statement = (
        select(predictions)
        .where(
            predictions.c.analysis_run_id
            == analysis_run_id
        )
        .order_by(
            predictions.c.telemetry_sample_id.asc()
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


def list_incidents(
    analysis_run_id: UUID,
) -> list[dict[str, Any]]:
    """
    Return incidents for one analysis run.
    """

    incidents = get_table(
        "incidents"
    )

    statement = (
        select(incidents)
        .where(
            incidents.c.analysis_run_id
            == analysis_run_id
        )
        .order_by(
            incidents.c.created_at.asc()
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

def get_latest_completed_analysis_run(
    session_id: UUID,
    model_version_id: UUID,
) -> dict[str, Any] | None:
    """
    Return the latest completed analysis run for
    a telemetry session and model version.
    """

    analysis_runs = get_table(
        "analysis_runs"
    )

    statement = (
        select(analysis_runs)
        .where(
            analysis_runs.c.session_id
            == session_id
        )
        .where(
            analysis_runs.c.model_version_id
            == model_version_id
        )
        .where(
            analysis_runs.c.status
            == "completed"
        )
        .order_by(
            analysis_runs
            .c
            .finished_at
            .desc()
            .nullslast(),
            analysis_runs
            .c
            .created_at
            .desc(),
        )
        .limit(1)
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


def get_prediction_risk_summary(
    analysis_run_id: UUID,
) -> dict[str, float | int]:
    """
    Return aggregate prediction statistics
    for one analysis run.
    """

    predictions = get_table(
        "predictions"
    )

    anomaly_count_expression = func.sum(
        case(
            (
                predictions
                .c
                .predicted_anomaly
                .is_(True),
                1,
            ),
            else_=0,
        )
    )

    statement = (
        select(
            func.count(
                predictions.c.id
            ).label(
                "total_predictions"
            ),
            func.coalesce(
                anomaly_count_expression,
                0,
            ).label(
                "total_anomalies"
            ),
            func.coalesce(
                func.avg(
                    predictions.c.risk_score
                ),
                0.0,
            ).label(
                "mean_risk_score"
            ),
            func.coalesce(
                func.max(
                    predictions.c.risk_score
                ),
                0.0,
            ).label(
                "maximum_risk_score"
            ),
        )
        .where(
            predictions.c.analysis_run_id
            == analysis_run_id
        )
    )

    with database_session() as session:
        row = (
            session.execute(statement)
            .mappings()
            .one()
        )

    return {
        "total_predictions": int(
            row["total_predictions"]
        ),
        "total_anomalies": int(
            row["total_anomalies"]
        ),
        "mean_risk_score": float(
            row["mean_risk_score"]
        ),
        "maximum_risk_score": float(
            row["maximum_risk_score"]
        ),
    }
def list_analysis_prediction_records(
    analysis_run_id: UUID,
    schema_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return persisted predictions joined with telemetry
    samples and feature vectors for one analysis run.
    """

    predictions = get_table(
        "predictions"
    )

    telemetry_samples = get_table(
        "telemetry_samples"
    )

    feature_vectors = get_table(
        "feature_vectors"
    )

    statement = (
        select(
            predictions.c.id.label(
                "prediction_id"
            ),
            predictions.c.analysis_run_id,
            predictions.c.telemetry_sample_id,
            predictions.c.predicted_anomaly,
            predictions.c.risk_level,
            predictions.c.risk_score,
            predictions.c.confidence_score,
            predictions.c.isolation_score,
            predictions.c.forecast_residual_score,
            predictions.c.rule_score,
            predictions.c.persistence_score,
            predictions.c.early_warning_score,
            predictions.c.top_feature,
            predictions.c.out_of_distribution,
            predictions.c.human_review_required,
            predictions.c.explanation,
            predictions.c.feature_contributions,
            predictions.c.rule_violations,
            predictions.c.affected_subsystems,
            predictions.c.prediction_metadata,
            telemetry_samples.c.sample_index,
            telemetry_samples.c.timestamp,
            telemetry_samples.c.segment_identifier,
            telemetry_samples.c.ground_truth_label,
            telemetry_samples.c.anomaly_type,
            telemetry_samples.c.sample_metadata,
            feature_vectors.c.schema_name,
            feature_vectors.c.schema_version,
            feature_vectors.c.feature_values,
        )
        .join(
            telemetry_samples,
            telemetry_samples.c.id
            == predictions.c.telemetry_sample_id,
        )
        .join(
            feature_vectors,
            feature_vectors.c.telemetry_sample_id
            == telemetry_samples.c.id,
        )
        .where(
            predictions.c.analysis_run_id
            == analysis_run_id
        )
    )

    if schema_name is not None:
        statement = statement.where(
            feature_vectors.c.schema_name
            == schema_name
        )

    statement = statement.order_by(
        telemetry_samples.c.sample_index.asc()
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


def update_incident_review(
    incident_id: UUID,
    status: str,
    operator_name: str,
    operator_note: str,
) -> dict[str, Any]:
    """
    Update an incident review status and preserve
    the complete operator-review history.
    """

    if status not in VALID_INCIDENT_STATUSES:
        raise ValueError(
            f"Invalid incident status: {status}"
        )

    clean_operator_name = operator_name.strip()
    clean_operator_note = operator_note.strip()

    if not clean_operator_name:
        raise ValueError(
            "Operator name cannot be empty."
        )

    if not clean_operator_note:
        raise ValueError(
            "Operator note cannot be empty."
        )

    incidents = get_table(
        "incidents"
    )

    with database_session() as session:
        existing_statement = (
            select(
                incidents.c.id,
                incidents.c.status,
                incidents.c.metadata,
            )
            .where(
                incidents.c.id
                == incident_id
            )
            .with_for_update()
        )

        existing_incident = (
            session.execute(
                existing_statement
            )
            .mappings()
            .one_or_none()
        )

        if existing_incident is None:
            raise RuntimeError(
                "Incident was not found: "
                f"{incident_id}"
            )

        previous_status = str(
            existing_incident["status"]
        )

        existing_metadata = (
            existing_incident["metadata"]
        )

        if isinstance(
            existing_metadata,
            dict,
        ):
            incident_metadata = dict(
                existing_metadata
            )
        else:
            incident_metadata = {}

        existing_history = (
            incident_metadata.get(
                "review_history",
                [],
            )
        )

        if isinstance(
            existing_history,
            list,
        ):
            review_history = list(
                existing_history
            )
        else:
            review_history = []

        reviewed_at = datetime.now(
            timezone.utc
        ).isoformat()

        review_entry = {
            "previous_status": previous_status,
            "new_status": status,
            "operator_name": clean_operator_name,
            "operator_note": clean_operator_note,
            "reviewed_at": reviewed_at,
        }

        review_history.append(
            review_entry
        )

        updated_metadata = {
            **incident_metadata,
            "operator_name": clean_operator_name,
            "operator_note": clean_operator_note,
            "last_reviewed_at": reviewed_at,
            "review_history": review_history,
        }

        update_statement = (
            update(incidents)
            .where(
                incidents.c.id
                == incident_id
            )
            .values(
                status=status,
                metadata=updated_metadata,
                updated_at=func.now(),
            )
            .returning(
                *incidents.c
            )
        )

        updated_incident = (
            session.execute(
                update_statement
            )
            .mappings()
            .one()
        )

    return dict(
        updated_incident
    )

def update_analysis_run_mission_health(
    analysis_run_id: UUID,
    mission_health_score: float,
    health_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    Persist the latest prototype mission-health snapshot.

    This is used after an analysis completes and after an
    operator changes an incident workflow status.
    """

    _validate_percentage(
        "mission_health_score",
        mission_health_score,
    )

    analysis_runs = get_table(
        "analysis_runs"
    )

    with database_session() as session:
        existing_statement = (
            select(
                analysis_runs.c.id,
                analysis_runs.c.metadata,
            )
            .where(
                analysis_runs.c.id
                == analysis_run_id
            )
            .with_for_update()
        )

        existing_run = (
            session.execute(existing_statement)
            .mappings()
            .one_or_none()
        )

        if existing_run is None:
            raise RuntimeError(
                "Analysis run was not found: "
                f"{analysis_run_id}"
            )

        existing_metadata = existing_run["metadata"]

        if isinstance(existing_metadata, dict):
            merged_metadata = dict(existing_metadata)
        else:
            merged_metadata = {}

        merged_metadata["mission_health"] = dict(
            health_snapshot
        )

        update_statement = (
            update(analysis_runs)
            .where(
                analysis_runs.c.id
                == analysis_run_id
            )
            .values(
                mission_health_score=float(
                    mission_health_score
                ),
                metadata=merged_metadata,
            )
            .returning(
                *analysis_runs.c
            )
        )

        updated_run = (
            session.execute(update_statement)
            .mappings()
            .one()
        )

    return dict(updated_run)
