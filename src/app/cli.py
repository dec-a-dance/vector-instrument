import argparse
import json
import sys

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


def build_parser():
    parser = argparse.ArgumentParser(prog="vector-cli")
    parser.add_argument("--db", default="vector_storage.db", help="Path to storage database")

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--file", required=True)
    p_ingest.add_argument("--table", default=None)

    p_set = sub.add_parser("set-subjective")
    p_set.add_argument("--vector", required=True)

    sub.add_parser("get-subjective")
    sub.add_parser("stats")
    sub.add_parser("reset")

    p_knn = sub.add_parser("knn")
    p_knn.add_argument("--vector", required=True)
    p_knn.add_argument("--k", type=int, default=5)

    p_knn_f = sub.add_parser("knn-filtered")
    p_knn_f.add_argument("--vector", required=True)
    p_knn_f.add_argument("--k", type=int, default=5)
    p_knn_f.add_argument("--weight", type=float, default=0.5)
    p_knn_f.add_argument("--threshold", type=float, default=None)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    storage = Storage(db_path=args.db)

    try:
        if args.command == "ingest":
            result = ingest_file(args.file, storage, args.table)
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
            result = knn_search(storage, vector, args.k)
            print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))

        elif args.command == "knn-filtered":
            vector = parse_vector(args.vector)
            knn_res = knn_search(storage, vector, args.k)
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