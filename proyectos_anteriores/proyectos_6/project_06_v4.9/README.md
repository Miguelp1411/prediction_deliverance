# Hybrid Multi-DB Weekly Schedule Predictor

A production-ready system for predicting weekly robot schedules using:

- **Template retrieval** from historical weeks
- **Residual neural models** for count correction and time prediction
- **CP-SAT exact scheduler** for conflict-free agenda construction
- **Multi-database support** with canonical schema

## Architecture

```
Template Retrieval → Residual Occurrence (Δ counts) → Residual Temporal (top-k slots) → CP-SAT Solver → Schedule
```

## Quick Start

### Training

```bash
# Single database
python train.py --databases bases_datos/nexus_schedule_10years.json

# Multiple databases (global training)
python train.py --config configs/train_global.yaml

# Quick smoke test (2 epochs)
python train.py --databases bases_datos/aux_database.json --max-epochs 2
```

### Prediction

```bash
# Predict last week
python predict.py --database bases_datos/nexus_schedule_10years.json --explain

# Predict specific week
python predict.py --database bases_datos/nexus_schedule_10years.json --week-index 500 --output results.json
```

### Database Profiling

```bash
python -m data.profiling bases_datos/nexus_schedule_10years.json
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `configs/` | YAML configuration files |
| `data/` | Schema, adapters, registry, profiling, preprocessing |
| `retrieval/` | Template retriever and builder |
| `features/` | Feature engineering (occurrence, temporal, calendar) |
| `models/` | Neural models (occurrence residual, temporal residual) |
| `scheduler/` | CP-SAT solver with constraints |
| `training/` | Engine, losses, reporting |
| `inference/` | Prediction pipeline, explanation |
| `evaluation/` | Matching, metrics, baselines |
| `bases_datos/` | Database JSON files |
| `checkpoints/` | Model checkpoints |
| `reports/` | Training and profiling reports |

## Configuration

All hyperparameters are in YAML files (`configs/base.yaml`). Override per use-case:

- `configs/train_global.yaml` — multi-database global training
- `configs/train_local.yaml` — per-database fine-tuning
- `configs/inference.yaml` — inference settings

## Requirements

- Python 3.10+
- PyTorch 2.0+
- OR-Tools (`pip install ortools`)
- NumPy, Pandas, SciPy, PyYAML
