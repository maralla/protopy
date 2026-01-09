from __future__ import annotations

from protopy.proto3 import build_proto3_grammar


def main() -> None:
    g = build_proto3_grammar()
    print(f"productions: {len(g.productions)}")
    for i, p in enumerate(g.productions):
        print(f"{i:>3}: {p}")


if __name__ == "__main__":
    main()

