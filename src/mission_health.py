from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

MISSION_HEALTH_WEIGHTS = {
    "telemetry_stability": 0.30,
    "anomaly_control": 0.20,
    "peak_resilience": 0.25,
    "incident_readiness": 0.25,
}

SEVERITY_WEIGHTS = {
    "critical": 1.00,
    "warning": 0.65,
    "watch": 0.35,
    "normal": 0.00,
}

STATUS_MULTIPLIERS = {
    "open": 1.00,
    "under_review": 1.00,
    "confirmed": 1.25,
    "rejected": 0.00,
    "resolved": 0.00,
}


def calculate_mission_health_score(
    frame: pd.DataFrame,
    incidents: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """
    Calculate a transparent prototype mission-health index.

    The index combines telemetry stability, anomaly rate,
    peak-risk resilience, and unresolved incident burden.
    It is a decision-support indicator, not a certified
    spacecraft-health measurement.
    """

    if frame.empty:
        return {
            "score": 0.0,
            "status": "Unavailable",
            "telemetry_stability": 0.0,
            "anomaly_control": 0.0,
            "peak_resilience": 0.0,
            "incident_readiness": 0.0,
            "incidents_considered": len(incidents),
            "unresolved_incidents": 0,
        }

    required_columns = {
        "hybrid_score",
        "prediction",
    }

    missing_columns = sorted(
        required_columns.difference(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Mission-health input is missing columns: "
            + ", ".join(missing_columns)
        )

    hybrid_scores = pd.to_numeric(
        frame["hybrid_score"],
        errors="coerce",
    ).fillna(0.0)

    predictions = pd.to_numeric(
        frame["prediction"],
        errors="coerce",
    ).fillna(0).astype(int)

    mean_risk = float(hybrid_scores.mean())
    peak_risk = float(hybrid_scores.max())
    anomaly_rate = float((predictions == 1).mean() * 100.0)

    telemetry_stability = float(
        np.clip(100.0 - mean_risk, 0.0, 100.0)
    )
    anomaly_control = float(
        np.clip(100.0 - anomaly_rate, 0.0, 100.0)
    )
    peak_resilience = float(
        np.clip(100.0 - peak_risk, 0.0, 100.0)
    )

    incident_burden = 0.0
    unresolved_incidents = 0

    for incident in incidents:
        severity = str(
            incident.get("severity", "watch")
        ).strip().lower()
        status = str(
            incident.get("status", "open")
        ).strip().lower()

        status_multiplier = STATUS_MULTIPLIERS.get(
            status,
            1.00,
        )

        if status_multiplier > 0.0:
            unresolved_incidents += 1

        incident_burden += (
            SEVERITY_WEIGHTS.get(severity, 0.35)
            * status_multiplier
        )

    if incidents:
        normalized_burden = float(
            np.clip(
                incident_burden / max(len(incidents), 1),
                0.0,
                1.0,
            )
        )
        incident_readiness = float(
            100.0 * (1.0 - normalized_burden)
        )
    else:
        incident_readiness = 100.0

    score = float(
        np.clip(
            telemetry_stability
            * MISSION_HEALTH_WEIGHTS["telemetry_stability"]
            + anomaly_control
            * MISSION_HEALTH_WEIGHTS["anomaly_control"]
            + peak_resilience
            * MISSION_HEALTH_WEIGHTS["peak_resilience"]
            + incident_readiness
            * MISSION_HEALTH_WEIGHTS["incident_readiness"],
            0.0,
            100.0,
        )
    )

    if score >= 85.0:
        status = "Nominal"
    elif score >= 70.0:
        status = "Stable"
    elif score >= 50.0:
        status = "Degraded"
    else:
        status = "Critical"

    return {
        "score": score,
        "status": status,
        "telemetry_stability": telemetry_stability,
        "anomaly_control": anomaly_control,
        "peak_resilience": peak_resilience,
        "incident_readiness": incident_readiness,
        "incidents_considered": len(incidents),
        "unresolved_incidents": unresolved_incidents,
    }
