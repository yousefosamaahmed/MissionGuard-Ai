from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from database.connection import check_database_connection


def main() -> None:
    print("=" * 60)
    print("Testing MissionGuard PostgreSQL connection...")
    print("=" * 60)

    try:
        info = check_database_connection()

        print(f"Database: {info['database_name']}")
        print(f"Current schema: {info['schema_name']}")
        print(
            f"Configured schema: "
            f"{info['configured_schema']}"
        )
        print(f"Tables found: {info['table_count']}")
        print("Connection successful.")

        missing_tables = info[
            "missing_required_tables"
        ]

        if missing_tables:
            raise RuntimeError(
                "Missing required tables: "
                + ", ".join(missing_tables)
            )

        print(
            "Required MissionGuard tables verified: "
            f"{info['required_table_count']}"
        )

    except Exception as error:
        print("Database connection failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {error}")
        raise


if __name__ == "__main__":
    main()