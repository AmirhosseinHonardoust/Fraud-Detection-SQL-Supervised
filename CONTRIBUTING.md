# Contributing

## Setup

```bash
make install
# or: pip install -r requirements.txt -r requirements-dev.txt
```

Optional: `pre-commit install` to run ruff/black/mypy automatically on each
commit (same versions CI uses).

## Before opening a PR

Run the full quality gate locally — it's the same thing CI runs:

```bash
make gate
```

This runs, in order: `ruff check`, `black --check`, `mypy src`, and
`pytest` (coverage must stay at or above 90%).

To exercise the actual pipeline end-to-end:

```bash
make run
```

## Guidelines

- Keep changes small and focused; prefer the smallest diff that achieves the
  goal.
- Add or update tests for any behavior change. Refactors should keep the
  test suite green and, where practical, show that outputs are unchanged.
- Match existing style: typed function signatures, `pathlib.Path` for paths,
  flags overridable via environment variables where one already exists.
