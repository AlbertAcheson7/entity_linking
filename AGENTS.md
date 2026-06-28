# Agent Instructions

This repository is an ICD-10 to ICD-11 MMS entity-linking research workspace.

## Working Rules

- Do not modify source data under `data/source/`.
- Do not commit generated artifacts from `data/prepared/`, `chroma/`, `models/`, `logs/`, or large experiment outputs unless explicitly requested.
- Preserve reproducibility: config-driven behavior should stay in `configs/*.yaml`.
- Prefer small, focused changes. Do not rewrite the pipeline unless asked.
- Existing user changes may be present. Do not revert unrelated edits.

## Important Entry Points

- CLI: `src/icd_linker/cli.py`
- Data preparation: `src/icd_linker/prepare.py`
- Text views: `src/icd_linker/text_views.py`
- Retrieval backends: `src/icd_linker/retrieval.py`
- Model adapters: `src/icd_linker/models.py`
- Evaluation metrics: `src/icd_linker/metrics.py`
- Tests: `tests/test_core.py`

## Current Research Direction

The old smoke-test system is archived in `docs/smoke-test-notes.md`.

Current work is about validating a new model/retrieval framework:
- matrix retrieval backend
- multiple embedding adapters
- target view expansion: `name_text`, `context_text`, `path_text`
- query variants: name/context/path
- view-record ranking vs entity-level aggregation
- metrics for one-to-many mappings

## Verification

For code-only changes, run:

```bash
python -m unittest