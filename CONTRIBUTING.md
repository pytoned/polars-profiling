# Contributing

Thanks for your interest in improving **polars-profiling**!

## Development setup

```bash
git clone https://github.com/pytoned/polars-profiling
cd polars-profiling
pip install -e ".[test]"
```

## Running the tests

```bash
pytest tests/unit/
```

The suite covers type detection, the statistics backend, report rendering and
the main features. Please make sure it passes (and add tests for new behaviour)
before opening a pull request.

## Guidelines

- Keep the library **Polars-only** — no `pandas`, `pyspark`, or other
  dataframe engines in the runtime path.
- Match the existing code style (`pre-commit run --all-files` runs the
  formatters and linters).
- Open an issue first for larger changes so we can discuss the approach.

## Releasing

Releases are published to PyPI automatically when a `vX.Y.Z` tag is pushed
(see `.github/workflows/publish-pypi.yml`). Bump `VERSION`, draft a GitHub
release with the matching tag, and the workflow does the rest.
