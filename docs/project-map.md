# Project map and naming guide

This project keeps source code, experiments, data, and generated evidence in separate top-level folders. Keep those roles clear when adding files. Read [`../AGENTS.md`](../AGENTS.md), [`../readme.md`](../readme.md), and [`agent-handoff.md`](agent-handoff.md) before changing experiment behavior or recorded results.

## Folder map

| Path | Purpose | What belongs here |
|---|---|---|
| `client/` | React/Vite user interface | App source, static public assets, and frontend scripts |
| `server/` | FastAPI application | API entry point, server requirements, and legacy server code in `archive/` |
| `src/` | Reusable Python application and research modules | Data validation, evaluation, models, preprocessing, tracking, and utilities |
| `scratch/` | One-off or opt-in experiment runners and shared experiment helpers | Scripts that read inputs and save explicitly named outputs; no irreplaceable source data |
| `tests/` | Automated tests | Tests for stable APIs, schemas, and reusable Python behavior |
| `Dataset/` | Original local fish dataset | Preserve the original data and annotations; do not rename or regenerate automatically |
| `external_data/` | Approved external datasets and provenance | Preserve licenses, source URLs, checksums, and original archives |
| `models/` | Model checkpoints | Treat checkpoint files as user artifacts; never overwrite them during experiments |
| `outputs/` | Experiment results and derived data | Save new results under a new, descriptive name; preserve metadata and provenance |
| `notebooks/` | Original training notebooks | Preserve the originals; sanitize copies without overwriting them |
| `docs/` | Project guidance, protocols, plans, and research writing | Keep limitations and experiment status synchronized with the artifacts |

## Naming conventions

- Use `snake_case.py` for Python modules and experiment scripts.
- Name an experiment runner `run_<task>.py`; state the dataset or split in its output artifact name when relevant.
- Name result files `<experiment>_<dataset-or-split>.<ext>`, for example `heldout_yolov8_metrics.json` and `enhancement_accuracy_test_clean.json`.
- Use descriptive, lowercase hyphenated names for Markdown documents, such as `video-tracking-validation.md`.
- Keep original dataset folder names, model checkpoint names, notebooks, and existing artifact paths stable. Scripts, tests, and docs refer to these paths.
- Do not combine metrics from separate runs into a new file without preserving each run's metadata and provenance.

## Safe organization rules

1. Prefer a small index or documentation update over moving an established file when the new name would break imports, scripts, API paths, or deployment configuration.
2. Before moving or renaming a source file, search the whole repository for its path and update every reference in one change. Keep a compatibility shim if external consumers may depend on the old path.
3. Do not move, rename, delete, or regenerate datasets, checkpoints, original notebooks, saved result JSON, or research PDFs without explicit approval.
4. Keep comments focused on why a non-obvious rule exists or what a result means; let clear names and functions explain routine operations.
5. Treat `client/dist/`, Python caches, virtual environments, and generated derived data as build/runtime artifacts. Do not commit them unless deployment instructions explicitly require them.

## First files to read

1. [`../AGENTS.md`](../AGENTS.md) for shared agent rules and verified project facts.
2. [`../readme.md`](../readme.md) for methods, evidence, results, and limitations.
3. [`agent-handoff.md`](agent-handoff.md) for the current implementation state and blockers.
4. The relevant module under `src/`, its experiment runner under `scratch/`, and related tests under `tests/` before changing behavior.
