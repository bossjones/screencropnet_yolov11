# Documentation

This directory contains the project documentation. Each file covers a distinct topic:

| File | Title | Description |
|------|-------|-------------|
| [quickstart.md](quickstart.md) | Quickstart | Get the async classify pipeline (FastAPI + worker + Postgres + CLI) running end to end |
| [screencrop-pipeline.md](screencrop-pipeline.md) | Screenshot ingest/classify pipeline | Architecture, endpoints, Prometheus metrics, export semantics, and the `SCREENCROPNET_` config reference |
| [installation.md](installation.md) | Installation | Installing uv and Python; quick setup cheat sheet for macOS/Linux |
| [usage.md](usage.md) | Usage | Training CLI syntax, options, and common Python API examples |
| [demo.md](demo.md) | Demo tool | Quick visual smoke test: sample images, run inference, and open an annotated contact-sheet montage |
| [dataset-preparation.md](dataset-preparation.md) | Dataset preparation | YOLO-format layout and four ingestion methods: validation, Roboflow pull, Pascal-VOC CSV conversion, and Label Studio |
| [configuration.md](configuration.md) | Configuration reference | All YAML training config keys, their defaults, and descriptions |
| [architecture.md](architecture.md) | Architecture | Pipeline stage breakdown and data-flow from dataset acquisition through model export |
| [api-reference.md](api-reference.md) | API reference | Public classes and functions for each module (`dataset_utils`, `model`, `training`, `evaluation`, `inference`, `visualization`) |
| [development.md](development.md) | Development | Fork/clone workflow, uv setup, and Makefile shortcuts for contributors |
| [label-studio-annotation-guide.md](label-studio-annotation-guide.md) | Label Studio annotation guide | Step-by-step bounding-box annotation of Twitter screenshots with EfficientNet-B0 ML backend pre-prediction |
| [adding-images-to-label-studio.md](adding-images-to-label-studio.md) | Adding images to Label Studio | Rename and stage new screenshots for annotation; Case A (no labels) and Case B (Pascal-VOC CSV) |
| [publishing.md](publishing.md) | Publishing releases | Releasing to PyPI via GitHub Actions with dynamic versioning |
