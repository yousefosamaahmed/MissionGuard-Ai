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


from database.repositories.missions import (
    create_mission,
    get_mission,
    list_missions,
)


def main() -> None:
    print("=" * 60)
    print("MissionGuard Database CRUD Test")
    print("=" * 60)

    unique_code = (
        "LOCAL-TEST-"
        + uuid4().hex[:8].upper()
    )

    print(
        "\nCreating a new mission..."
    )

    mission_id = create_mission(
        name=(
            "MissionGuard OPS-SAT "
            "Local Test"
        ),
        mission_code=unique_code,
        spacecraft_name="ESA OPS-SAT",
        description=(
            "Local PostgreSQL repository "
            "integration test."
        ),
        status="active",
    )

    print(
        f"Created mission ID: "
        f"{mission_id}"
    )

    print(
        "\nReading the created mission..."
    )

    mission = get_mission(
        mission_id
    )

    if mission is None:
        raise RuntimeError(
            "The created mission could not be found."
        )

    print(
        f"Mission name: "
        f"{mission['name']}"
    )

    print(
        f"Mission code: "
        f"{mission['mission_code']}"
    )

    print(
        f"Mission status: "
        f"{mission['status']}"
    )

    print(
        "\nReading all missions..."
    )

    missions = list_missions()

    print(
        f"Stored missions: "
        f"{len(missions)}"
    )

    for stored_mission in missions:
        print(
            "-",
            stored_mission["id"],
            "|",
            stored_mission["name"],
            "|",
            stored_mission["status"],
        )

    print(
        "\nCRUD test completed successfully."
    )


if __name__ == "__main__":
    main()