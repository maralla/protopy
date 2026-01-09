from __future__ import annotations

import argparse
from pathlib import Path

from protopy.testing import generate_corpus_files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="generate_corpus")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out", default="tests/fixtures/generated_corpus")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).resolve() / f"seed_{args.seed}_count_{args.count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rel, src in generate_corpus_files(seed=args.seed, count=args.count):
        p = out_dir / rel
        p.write_text(src, encoding="utf-8")

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

