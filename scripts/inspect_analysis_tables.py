from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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

TABLE_NAMES: tuple[str, ...] = (
    "model_versions",
    "model_artifacts",
    "model_metrics",
    "analysis_runs",
    "predictions",
    "incidents",
)


def print_columns(
    inspector: Any,
    table_name: str,
) -> None:
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


def print_primary_key(
    inspector: Any,
    table_name: str,
) -> None:
    primary_key = inspector.get_pk_constraint(
        table_name,
        schema=POSTGRES_SCHEMA,
    )

    constrained_columns = primary_key.get(
        "constrained_columns",
        [],
    )

    print("\nPRIMARY KEY:")

    if constrained_columns:
        print(
            "- "
            + ", ".join(constrained_columns)
        )

    else:
        print("- No primary key found.")


def print_foreign_keys(
    inspector: Any,
    table_name: str,
) -> None:
    foreign_keys = inspector.get_foreign_keys(
        table_name,
        schema=POSTGRES_SCHEMA,
    )

    print("\nFOREIGN KEYS:")

    if not foreign_keys:
        print("- No foreign keys found.")
        return

    for foreign_key in foreign_keys:
        constrained_columns = foreign_key.get(
            "constrained_columns",
            [],
        )

        referred_schema = foreign_key.get(
            "referred_schema",
        )

        referred_table = foreign_key.get(
            "referred_table",
        )

        referred_columns = foreign_key.get(
            "referred_columns",
            [],
        )

        print(
            f"- {constrained_columns}"
            f" -> "
            f"{referred_schema}."
            f"{referred_table}"
            f"{referred_columns}"
        )


def print_unique_constraints(
    inspector: Any,
    table_name: str,
) -> None:
    constraints = inspector.get_unique_constraints(
        table_name,
        schema=POSTGRES_SCHEMA,
    )

    print("\nUNIQUE CONSTRAINTS:")

    if not constraints:
        print("- No unique constraints found.")
        return

    for constraint in constraints:
        print(
            f"- name={constraint.get('name')!r}"
            f" | columns="
            f"{constraint.get('column_names', [])}"
        )


def print_check_constraints(
    inspector: Any,
    table_name: str,
) -> None:
    constraints = inspector.get_check_constraints(
        table_name,
        schema=POSTGRES_SCHEMA,
    )

    print("\nCHECK CONSTRAINTS:")

    if not constraints:
        print("- No check constraints found.")
        return

    for constraint in constraints:
        print(
            f"- name={constraint.get('name')!r}"
            f" | sqltext="
            f"{constraint.get('sqltext')!r}"
        )


def print_indexes(
    inspector: Any,
    table_name: str,
) -> None:
    indexes = inspector.get_indexes(
        table_name,
        schema=POSTGRES_SCHEMA,
    )

    print("\nINDEXES:")

    if not indexes:
        print("- No indexes found.")
        return

    for index in indexes:
        print(
            f"- name={index.get('name')!r}"
            f" | columns={index.get('column_names', [])}"
            f" | unique={index.get('unique', False)}"
        )


def inspect_table(
    inspector: Any,
    table_name: str,
) -> None:
    print("\n" + "=" * 90)
    print(
        f"TABLE: {POSTGRES_SCHEMA}.{table_name}"
    )
    print("=" * 90)

    table_exists = inspector.has_table(
        table_name,
        schema=POSTGRES_SCHEMA,
    )

    if not table_exists:
        print("ERROR: Table does not exist.")
        return

    print_columns(
        inspector,
        table_name,
    )

    print_primary_key(
        inspector,
        table_name,
    )

    print_foreign_keys(
        inspector,
        table_name,
    )

    print_unique_constraints(
        inspector,
        table_name,
    )

    print_check_constraints(
        inspector,
        table_name,
    )

    print_indexes(
        inspector,
        table_name,
    )


def main() -> None:
    print("=" * 90)
    print("MissionGuard Model and Analysis Tables Inspection")
    print("=" * 90)

    print(
        f"PostgreSQL schema: "
        f"{POSTGRES_SCHEMA}"
    )

    inspector = inspect(engine)

    for table_name in TABLE_NAMES:
        inspect_table(
            inspector,
            table_name,
        )

    print("\n" + "=" * 90)
    print("Inspection completed.")
    print("=" * 90)


if __name__ == "__main__":
    main()