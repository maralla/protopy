# Agent Guidelines

This document contains guidelines and rules for AI agents working on this codebase.

## Type Checking Rules

### Never Ignore Type Errors

**Never** add comments like `# type: ignore` to bypass type checker issues.

Instead of ignoring type errors:

1. **Fix the underlying issue** - Understand why the type checker is complaining and address the root cause
2. **Add proper type annotations** - Ensure all functions, classes, and variables have correct type hints
4. **Use `cast()` only when necessary** - If you must use `cast()`, ensure it's truly needed and document why with a comment explaining the reasoning

### Examples

**Bad:**
```python
def foo(x: Any) -> str:
    return x.upper()  # type: ignore[attr-defined]
```

**Good:**
```python
def foo(x: str) -> str:
    return x.upper()
```

**Bad:**
```python
result = some_function()  # type: ignore
```

**Good:**
```python
result: ExpectedType = some_function()
# or fix some_function's return type annotation
```

## Code Quality Standards

### Linting

- Always run `uv run ruff check` and fix all errors
- Use `uv run ruff check --fix` for auto-fixable issues
- Do not use `# noqa` comments to suppress linter warnings without good reason
- If complexity warnings appear (C901, PLR0912), refactor the code to reduce complexity rather than suppressing the warning

### Type Checking

- Always run `uv run mypy` and fix all errors
- Maintain strict type checking throughout the codebase

## General Principles

1. **Fix, don't suppress** - Always prefer fixing issues over suppressing warnings
2. **Understand before acting** - Don't blindly apply fixes; understand what the error means
3. **Maintain consistency** - Follow existing patterns and conventions in the codebase
4. **Test your changes** - Ensure type checking and linting pass after every change
