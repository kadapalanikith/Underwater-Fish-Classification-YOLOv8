# Project map and naming guide

This project keeps source code, experiments, data, and generated evidence in clearly named folders. Keep those roles clear when adding files. Read [`../AGENTS.md`](../AGENTS.md), [`../README.md`](../README.md), and [`agent-handoff.md`](agent-handoff.md) before changing experiment behavior or recorded results.

## Folder map

| Path | Purpose | What belongs here |
|---|---|---|
| `client/` | React/Vite user interface | App source, static public assets, and frontend scripts |
| `server/` | FastAPI application | API entry point, server requirements, and legacy server code in `archive/` |
| `src/` | Reusable Python application and research modules | Data validation, evaluation, models, preprocessing, tracking, and utilities |
| `scratch/` | One-off or opt-in experiment runners and shared experiment helpers | Scripts that read inputs and save explicitly named outputs; no irreplaceable source data |
| `tests/` | Automated tests | Tests for stable APIs, schemas, and reusable Python behavior |
| `Dataset/` | Original local fish dataset | Preserve the original data and annotations; do not rename or regenerate automatically |
| `qualitative_samples/` | Unlabeled example images | Use for qualitative smoke checks only; not an independent test or cross-domain dataset |
| `external_data/` | Approved external datasets and provenance | Preserve licenses, source URLs, checksums, and original archives |
| `models/` | Model checkpoints | Treat checkpoint files as user artifacts; never overwrite them during experiments |
| `outputs/` | Experiment results and derived data | Save new results under a new, descriptive name; preserve metadata and provenance |
| `notebooks/training/` | Colab training workflows | Descriptively named notebooks; credentials must come from environment variables |
| `docs/` | Project, code, deployment, research, and experiment documentation | Keep limitations and experiment status synchronized with artifacts |

## Naming conventions

- Use `snake_case.py` for Python modules and experiment scripts.
- Name an experiment runner `run_<task>.py`; state the dataset or split in its output artifact name when relevant.
- Name result files `<experiment>_<dataset-or-split>.<ext>`, for example `heldout_yolov8_metrics.json` and `enhancement_accuracy_test_clean.json`.
- Use descriptive, lowercase hyphenated names for Markdown documents, such as `video-tracking-validation.md`.
- Keep original dataset folder names, model checkpoint names, and established artifact paths stable. Store notebooks under the relevant `notebooks/` subfolder and use descriptive names; update references after an approved rename.
- Do not combine metrics from separate runs into a new file without preserving each run's metadata and provenance.

## Safe organization rules

1. Prefer a small index or documentation update over moving an established file when the new name would break imports, scripts, API paths, or deployment configuration.
2. Before moving or renaming a source file, search the whole repository for its path and update every reference in one change. Keep a compatibility shim if external consumers may depend on the old path.
3. Do not move, rename, delete, or regenerate datasets, checkpoints, original notebooks, saved result JSON, or research PDFs without explicit approval.
4. Keep comments focused on why a non-obvious rule exists or what a result means; let clear names and functions explain routine operations.
5. Treat `client/dist/`, Python caches, virtual environments, and generated derived data as build/runtime artifacts. Do not commit them unless deployment instructions explicitly require them.

## Notebook index

The two Colab notebooks follow the same fish-detection training workflow. `fish_detection_training.ipynb` has no execution counters but retains some saved outputs; `fish_detection_training_executed.ipynb` contains execution metadata and outputs. Both read `KAGGLE_USERNAME` and `KAGGLE_KEY` from the runtime environment to create the local Kaggle config file. Never store credentials in notebook cells. See [`../notebooks/README.md`](../notebooks/README.md).

## First files to read

1. [`../AGENTS.md`](../AGENTS.md) for shared agent rules and verified project facts.
2. [`../README.md`](../README.md) for methods, evidence, results, and limitations.
3. [`agent-handoff.md`](agent-handoff.md) for the current implementation state and blockers.
4. The relevant module under `src/`, its experiment runner under `scratch/`, and related tests under `tests/` before changing behavior.
