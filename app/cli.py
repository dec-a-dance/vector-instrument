import argparse
import json
import sys
from typing import Dict, Any

from .filter import filter_by_subjective_dynamics
from .ingest import ingest_file
from .knn import knn_search
from .storage import Storage
from .vectors import get_subjective_vector, set_subjective_vector


def parse_vector(raw: str):
    value = json.loads(raw)
    if isinstance(value, list):
        return [float(x) for x in value]
    raise ValueError("Vector must be a JSON array")


def parse_list(raw: str):
    """Парсит строку вида 'col1,col2' или JSON-список в List[str]"""
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return [str(x) for x in value]
    except json.JSONDecodeError:
        pass
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_filters(raw_filters):
    """
    Парсит фильтры вида:
      --filter "category=books"
      --filter "region=eu"
    """
    if not raw_filters:
        return {}

    result: Dict[str, Any] = {}
    for raw in raw_filters:
        if "=" not in raw:
            raise ValueError(f"Invalid filter format: {raw}. Use key=value")

        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(f"Invalid filter key in: {raw}")

        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value

        result[key] = parsed_value

    return result


def build_parser():
    parser = argparse.ArgumentParser(prog="vector-cli")
    parser.add_argument("--db", default="vector_storage.db", help="Path to storage database")

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--file", required=True, help="Path to CSV/Excel file")
    p_ingest.add_argument("--table", default=None, help="Target table name")
    p_ingest.add_argument(
        "--label-cols",
        default=None,
        help="Comma-separated list of non-numeric columns to keep as labels (e.g. 'id,name')"
    )

    p_set = sub.add_parser("set-subjective")
    p_set.add_argument("--vector", required=True)

    sub.add_parser("get-subjective")
    sub.add_parser("stats")
    sub.add_parser("reset")

    p_knn = sub.add_parser("knn")
    p_knn.add_argument("--vector", required=True)
    p_knn.add_argument("--k", type=int, default=5)
    p_knn.add_argument(
        "--filter",
        action="append",
        default=None,
        help="Metadata filter in key=value format. Can be repeated."
    )

    p_knn_f = sub.add_parser("knn-filtered")
    p_knn_f.add_argument("--vector", required=True)
    p_knn_f.add_argument("--k", type=int, default=5)
    p_knn_f.add_argument("--weight", type=float, default=0.5)
    p_knn_f.add_argument("--threshold", type=float, default=None)
    p_knn_f.add_argument(
        "--filter",
        action="append",
        default=None,
        help="Metadata filter in key=value format. Can be repeated."
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    storage = Storage(db_path=args.db)

    try:
        if args.command == "ingest":
            label_columns = parse_list(args.label_cols) if args.label_cols else []

            result = ingest_file(
                file_path=args.file,
                storage=storage,
                table_name=args.table,
                keep_columns=label_columns
            )
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))

        elif args.command == "set-subjective":
            vector = parse_vector(args.vector)
            set_subjective_vector(storage, vector)
            print(json.dumps({"ok": True}, ensure_ascii=False))

        elif args.command == "get-subjective":
            result = get_subjective_vector(storage)
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))

        elif args.command == "stats":
            print(json.dumps({"ok": True, "result": storage.stats()}, ensure_ascii=False))

        elif args.command == "reset":
            storage.reset_all()
            print(json.dumps({"ok": True}, ensure_ascii=False))

        elif args.command == "knn":
            vector = parse_vector(args.vector)
            filters = parse_filters(args.filter)
            result = knn_search(storage, vector, args.k, filters=filters or None)
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))

        elif args.command == "knn-filtered":
            vector = parse_vector(args.vector)
            filters = parse_filters(args.filter)
            knn_res = knn_search(storage, vector, args.k, filters=filters or None)
            subj = storage.get_subjective_vector()

            result = filter_by_subjective_dynamics(
                knn_res,
                subj or [],
                {"weight": args.weight, "threshold": args.threshold}
            )
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))

        else:
            raise ValueError("Unknown command")

    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()