# Entity Linking

This repository contains the code workspace for Medical Terminology Mapping experiments.

## Current Goal
Replace the previous smoke-test implementation with a new model framework for experimental validation.

## Previous Work
- Basic embedding-based dense retrieval baseline has been completed.
- Previous smoke-test notes are archived in `docs/smoke-test-notes.md`.

## Main Tasks
- Implement the new model framework.
- Keep baseline evaluation reproducible.
- Compare new models against the previous dense retrieval baseline.
- Record experiment results and error analysis.

## Project Layout
- `data`: source data and target data
- `src/`: model and training/evaluation code
- `configs/`: experiment configs
- `scripts/`: runnable commands
- `experiments/`: metrics, predictions, and analysis outputs
- `docs/`: design notes and experiment records