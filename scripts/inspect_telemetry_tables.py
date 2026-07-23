from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import inspect

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from database.connection import (
    POSTGRES_SCHEMA,
    engine,
)

TABLE_NAMES = [
    "telemetry_samples",
    "feature_vectors",
    "data_quality_reports",
]


def print_table_information(
    table_name: str,
) -> None:
    """
    Print columns, primary keys and foreign keys
    for an existing PostgreSQL table.
    """

    inspector = inspect(engine)

    print("\n" + "=" * 80)
    print(
        f"TABLE: {POSTGRES_SCHEMA}.{table_name}"
    )
    print("=" * 80)

    table_exists = inspector.has_table(
        table_name,
        schema=POSTGRES_SCHEMA,
    )

    if not table_exists:
        print("ERROR: Table does not exist.")
        return

    columns = inspector.get_columns(
        table_name,
        schema=POSTGRES_SCHEMA,
    )

    print("\nCOLUMNS:")

    for column in columns:
        print(
            f"- name={column['name']!r}"
            f" | type={column['type']}"
            f" | nullable={column['nullable']}"
            f" | default={column.get('default')!r}"
        )

    primary_key = inspector.get_pk_constraint(
        table_name,
        schema=POSTGRES_SCHEMA,
    )

    print("\nPRIMARY KEY:")

    primary_key_columns = primary_key.get(
        "constrained_columns",
        [],
    )

    if primary_key_columns:
        print(
            "- "
            + ", ".join(primary_key_columns)
        )

    else:
        print("- No primary key found.")

    foreign_keys = inspector.get_foreign_keys(
        table_name,
        schema=POSTGRES_SCHEMA,
    )

    print("\nFOREIGN KEYS:")

    if not foreign_keys:
        print("- No foreign keys found.")

    for foreign_key in foreign_keys:
        local_columns = foreign_key.get(
            "constrained_columns",
            [],
        )

        referenced_schema = foreign_key.get(
            "referred_schema",
        )

        referenced_table = foreign_key.get(
            "referred_table",
        )

        referenced_columns = foreign_key.get(
            "referred_columns",
            [],
        )

        print(
            f"- {local_columns}"
            f" -> "
            f"{referenced_schema}."
            f"{referenced_table}"
            f"{referenced_columns}"
        )

    unique_constraints = (
        inspector.get_unique_constraints(
            table_name,
            schema=POSTGRES_SCHEMA,
        )
    )

    print("\nUNIQUE CONSTRAINTS:")

    if not unique_constraints:
        print("- No unique constraints found.")

    for constraint in unique_constraints:
        print(
            f"- name={constraint.get('name')!r}"
            f" | columns="
            f"{constraint.get('column_names', [])}"
        )


def main() -> None:
    print("=" * 80)
    print("MissionGuard Telemetry Tables Inspection")
    print("=" * 80)

    print(
        f"PostgreSQL schema: "
        f"{POSTGRES_SCHEMA}"
    )

    for table_name in TABLE_NAMES:
        print_table_information(
            table_name
        )

    print("\n" + "=" * 80)
    print("Inspection completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()