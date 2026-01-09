# protopy

`protopy` is a **proto3-only** `.proto` parser that:

- Tokenizes and parses protobuf source **without depending on `protoc`**.
- Uses a **from-scratch LALR(1)** parser (tables generated from an explicit grammar).
- Produces a **custom AST** designed for downstream tooling (linters, generators, indexers).
- Supports parsing a **set of files** with **import resolution**.
- Reports **user-friendly errors** with precise file/line/column locations.

## Install (dev)

This project uses [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
uv run pytest
```

## Quickstart

```python
from protopy import parse_files

result = parse_files(
    entrypoints=["/abs/path/to/root.proto"],
    import_paths=["/abs/path/to/include"],
)

# result.files maps absolute path -> AST File node
root_ast = result.files[result.entrypoints[0]]
```

## Scope

- **Supported**: proto3 syntax (messages, enums, services, options, imports, packages, oneof,
  fields, map fields, reserved, nested types, RPC definitions, literals, comments).
- **Not supported (for now)**: proto2-only constructs (e.g. `required`, `extensions`),
  custom options beyond syntactic parsing, edition-specific features.

