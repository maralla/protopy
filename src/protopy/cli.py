from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass

from .api import parse_files


def _to_jsonable(obj):
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, tuple):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="protopy", description="Parse proto3 .proto files")
    ap.add_argument("entrypoints", nargs="+", help="Entry .proto files")
    ap.add_argument(
        "-I",
        "--import-path",
        action="append",
        default=[],
        help="Additional import root (repeatable)",
    )
    ap.add_argument("--json", action="store_true", help="Print parsed AST as JSON")
    args = ap.parse_args(argv)

    res = parse_files(entrypoints=args.entrypoints, import_paths=args.import_path)
    if args.json:
        payload = {
            "entrypoints": list(res.entrypoints),
            "files": {k: _to_jsonable(v) for k, v in res.files.items()},
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for p in res.entrypoints:
            print(p)
    return 0

