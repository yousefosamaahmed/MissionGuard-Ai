from __future__ import annotations

import pandas as pd
import pytest

from src.mission_health import calculate_mission_health_score


def test_empty_input_is_unavailable() -> None:
    result = calculate_mission_health_score(
        pd.DataFrame(),
        [],
    )

    assert result["score"] == 0.0
    assert result["status"] == "Unavailable"


def test_nominal_data_without_incidents_scores_100() -> None:
    frame = pd.DataFrame(
        {
            "hybrid_score": [0.0, 0.0, 0.0],
            "prediction": [0, 0, 0],
        }
    )

    result = calculate_mission_health_score(frame, [])

    assert result["score"] == pytest.approx(100.0)
    assert result["status"] == "Nominal"
    assert result["incident_readiness"] == pytest.approx(100.0)


def test_open_critical_incident_reduces_readiness() -> None:
    frame = pd.DataFrame(
        {
            "hybrid_score": [10.0, 20.0],
            "prediction": [0, 1],
        }
    )

    result = calculate_mission_health_score(
        frame,
        [{"severity": "Critical", "status": "open"}],
    )

    assert result["incident_readiness"] == pytest.approx(0.0)
    assert result["unresolved_incidents"] == 1


def test_resolved_and_rejected_incidents_do_not_add_burden() -> None:
    frame = pd.DataFrame(
        {
            "hybrid_score": [10.0, 20.0],
            "prediction": [0, 1],
        }
    )

    result = calculate_mission_health_score(
        frame,
        [
            {"severity": "Critical", "status": "resolved"},
            {"severity": "Warning", "status": "rejected"},
        ],
    )

    assert result["incident_readiness"] == pytest.approx(100.0)
    assert result["unresolved_incidents"] == 0


def test_missing_required_columns_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        calculate_mission_health_score(
            pd.DataFrame({"hybrid_score": [10.0]}),
            [],
        )
