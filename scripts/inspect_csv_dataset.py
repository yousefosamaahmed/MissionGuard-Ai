from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_csv_files() -> list[Path]:
    """
    Find CSV files inside the project data directory.
    """

    data_directory = PROJECT_ROOT / "data"

    if not data_directory.exists():
        return []

    return sorted(
        data_directory.rglob("*.csv")
    )


def inspect_csv(file_path: Path) -> None:
    """
    Inspect a CSV file without changing its contents.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file does not exist: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            "The selected file must be a CSV file."
        )

    dataframe = pd.read_csv(
        file_path,
        low_memory=False,
    )

    print("=" * 80)
    print("MissionGuard CSV Inspection")
    print("=" * 80)

    print(f"File: {file_path}")
    print(f"File size: {file_path.stat().st_size} bytes")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")

    print("\nColumn names:")

    for index, column_name in enumerate(
        dataframe.columns,
        start=1,
    ):
        print(
            f"{index}. {column_name!r}"
        )

    print("\nData types:")

    for column_name, data_type in dataframe.dtypes.items():
        print(
            f"- {column_name!r}: {data_type}"
        )

    print("\nMissing values:")

    missing_values = (
        dataframe
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_values_found = False

    for column_name, missing_count in missing_values.items():
        missing_count_value = int(
            missing_count
        )

        if missing_count_value > 0:
            missing_values_found = True

            print(
                f"- {column_name!r}: "
                f"{missing_count_value}"
            )

    if not missing_values_found:
        print("- No missing values detected.")

    print("\nDuplicate rows:")

    duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    print(
        f"- {duplicate_rows}"
    )

    print("\nFirst 5 rows:")

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
    ):
        print(
            dataframe.head(5).to_string(
                index=False
            )
        )

    print("\nNumeric statistics:")

    numeric_dataframe = dataframe.select_dtypes(
        include="number"
    )

    if numeric_dataframe.empty:
        print("- No numeric columns detected.")

    else:
        with pd.option_context(
            "display.max_columns",
            None,
            "display.width",
            200,
        ):
            print(
                numeric_dataframe
                .describe()
                .transpose()
                .to_string()
            )

    print("\nPossible special columns:")

    lowered_columns = {
        str(column).lower(): str(column)
        for column in dataframe.columns
    }

    timestamp_candidates = [
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "created_at",
    ]

    label_candidates = [
        "ground_truth_label",
        "is_anomaly",
        "anomaly",
        "label",
        "target",
    ]

    anomaly_type_candidates = [
        "anomaly_type",
        "fault_type",
        "event_type",
        "class",
    ]

    detected_timestamp = next(
        (
            lowered_columns[candidate]
            for candidate in timestamp_candidates
            if candidate in lowered_columns
        ),
        None,
    )

    detected_label = next(
        (
            lowered_columns[candidate]
            for candidate in label_candidates
            if candidate in lowered_columns
        ),
        None,
    )

    detected_anomaly_type = next(
        (
            lowered_columns[candidate]
            for candidate in anomaly_type_candidates
            if candidate in lowered_columns
        ),
        None,
    )

    print(
        f"- Timestamp column: "
        f"{detected_timestamp!r}"
    )

    print(
        f"- Label column: "
        f"{detected_label!r}"
    )

    print(
        f"- Anomaly type column: "
        f"{detected_anomaly_type!r}"
    )

    print("\nInspection completed.")


def main() -> None:
    if len(sys.argv) < 2:
        csv_files = find_csv_files()

        print("=" * 80)
        print("CSV Files Found")
        print("=" * 80)

        if not csv_files:
            print(
                "No CSV files were found inside "
                f"{PROJECT_ROOT / 'data'}"
            )

            print(
                "\nRun the script with a file path:"
            )

            print(
                r'.\.venv\Scripts\python.exe '
                r'scripts\inspect_csv_dataset.py '
                r'"path\to\file.csv"'
            )

            return

        for file_number, csv_file in enumerate(
            csv_files,
            start=1,
        ):
            relative_path = csv_file.relative_to(
                PROJECT_ROOT
            )

            print(
                f"{file_number}. {relative_path}"
            )

        print(
            "\nRun the script again with one file path."
        )

        return

    supplied_path = Path(
        sys.argv[1]
    )

    if not supplied_path.is_absolute():
        supplied_path = (
            PROJECT_ROOT
            / supplied_path
        )

    inspect_csv(
        supplied_path.resolve()
    )


if __name__ == "__main__":
    main()