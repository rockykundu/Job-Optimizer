import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
AUDIT_DIR = PROJECT_ROOT / "data" / "audit"
REPORT_PATH = AUDIT_DIR / "role_radar_structure.txt"

DATASET_FILES = [
    "scraped_jobs.json",
    "synthetic_profiles.json",
    "phase3_pairs.json",
    "phase3_labels.json",
    "gold_labels.json",
    "benchmarks.json",
]


def load_json(file_path: Path) -> Any:
    with file_path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def shorten(value: Any, limit: int = 140) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ")

    if len(text) > limit:
        return text[:limit] + "..."

    return text


def value_description(value: Any) -> str:
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "boolean"

    if isinstance(value, int):
        return "integer"

    if isinstance(value, float):
        return "float"

    if isinstance(value, str):
        return f"string: {shorten(value)}"

    if isinstance(value, list):
        return f"list with {len(value)} items"

    if isinstance(value, dict):
        return f"object with {len(value)} fields"

    return type(value).__name__


def find_record_collection(data: Any):
    """
    Attempts to locate the principal list of records in each JSON file.
    """

    if isinstance(data, list):
        return data, "JSON root"

    if isinstance(data, dict):
        preferred_keys = [
            "data",
            "records",
            "jobs",
            "profiles",
            "pairs",
            "labels",
            "results",
            "benchmarks",
        ]

        for key in preferred_keys:
            value = data.get(key)

            if isinstance(value, list):
                return value, f"root['{key}']"

        list_fields = [
            (key, value)
            for key, value in data.items()
            if isinstance(value, list)
        ]

        if list_fields:
            largest_key, largest_list = max(
                list_fields,
                key=lambda item: len(item[1]),
            )
            return largest_list, f"root['{largest_key}']"

    return None, None


def inspect_records(records: list, report: list[str]) -> None:
    report.append(f"Record count: {len(records):,}")

    if not records:
        report.append("The record collection is empty.")
        return

    record_types = Counter(type(record).__name__ for record in records)

    report.append(f"Record types: {dict(record_types)}")

    dictionary_records = [
        record for record in records if isinstance(record, dict)
    ]

    if not dictionary_records:
        report.append("Records are not JSON objects.")
        report.append(f"First value: {shorten(records[0])}")
        return

    all_keys = sorted(
        {
            key
            for record in dictionary_records
            for key in record.keys()
        }
    )

    report.append(f"Unique fields: {len(all_keys)}")
    report.append("Field coverage and detected value types:")

    for key in all_keys:
        present_count = sum(key in record for record in dictionary_records)
        null_count = sum(
            record.get(key) is None
            for record in dictionary_records
            if key in record
        )

        detected_types = sorted(
            {
                type(record[key]).__name__
                for record in dictionary_records
                if key in record and record[key] is not None
            }
        )

        report.append(
            f"  - {key}: present={present_count:,}/"
            f"{len(dictionary_records):,}, "
            f"null={null_count:,}, "
            f"types={detected_types}"
        )

    report.append("")
    report.append("First-record preview:")

    first_record = dictionary_records[0]

    for key, value in first_record.items():
        report.append(f"  - {key}: {value_description(value)}")

        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                report.append(
                    f"      - {nested_key}: "
                    f"{value_description(nested_value)}"
                )

        elif isinstance(value, list) and value:
            first_item = value[0]
            report.append(
                f"      First list item: "
                f"{value_description(first_item)}"
            )

            if isinstance(first_item, dict):
                for nested_key, nested_value in first_item.items():
                    report.append(
                        f"          - {nested_key}: "
                        f"{value_description(nested_value)}"
                    )


def inspect_file(file_path: Path) -> list[str]:
    report = [
        "=" * 80,
        f"FILE: {file_path.name}",
        f"SIZE: {file_path.stat().st_size:,} bytes",
    ]

    try:
        data = load_json(file_path)
    except json.JSONDecodeError as error:
        report.append(f"INVALID JSON: {error}")
        return report
    except Exception as error:
        report.append(f"FAILED TO READ FILE: {error}")
        return report

    report.append(f"Root JSON type: {type(data).__name__}")

    if isinstance(data, dict):
        report.append(f"Root fields: {sorted(data.keys())}")

    records, location = find_record_collection(data)

    if records is not None:
        report.append(f"Primary record collection: {location}")
        inspect_records(records, report)
    else:
        report.append(
            "No primary record list was automatically detected."
        )

        if isinstance(data, dict):
            report.append("Root-value descriptions:")

            for key, value in data.items():
                report.append(
                    f"  - {key}: {value_description(value)}"
                )

    return report


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    complete_report = [
        "ROLE RADAR DATASET — STRUCTURAL INSPECTION",
        f"Dataset directory: {RAW_DATA_DIR}",
        "",
    ]

    missing_files = []

    for filename in DATASET_FILES:
        file_path = RAW_DATA_DIR / filename

        if not file_path.exists():
            missing_files.append(filename)
            continue

        complete_report.extend(inspect_file(file_path))
        complete_report.append("")

    if missing_files:
        complete_report.append("=" * 80)
        complete_report.append("MISSING FILES:")
        complete_report.extend(
            f"  - {filename}" for filename in missing_files
        )

    REPORT_PATH.write_text(
        "\n".join(complete_report),
        encoding="utf-8",
    )

    print("Dataset inspection completed successfully.")
    print(f"Report created at: {REPORT_PATH}")

    if missing_files:
        print("Warning: Some expected files were missing.")
        for filename in missing_files:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()