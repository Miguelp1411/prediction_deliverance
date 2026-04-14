#!/usr/bin/env python3
"""
CLI: predict — run inference on a target week.

Usage:
    python predict.py --config configs/inference.yaml --database bases_datos/nexus_schedule_10years.json
    python predict.py --week-index 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from config import load_config, resolve_device
from data.adapters.json_adapter import load_json_events
from data.preprocessing import prepare_data
from data.profiling import profile_database
from data.schema import PreparedData
from features.occurrence_features import occurrence_feature_dim
from features.temporal_features import temporal_feature_dim
from inference.explain import explain_prediction
from inference.predict_week import predict_week
from models.occurrence_residual import OccurrenceResidualModel
from models.temporal_residual import TemporalResidualModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict weekly schedule")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--database", type=str, required=True, help="Database JSON path")
    parser.add_argument("--db-id", type=str, default=None, help="Database ID")
    parser.add_argument("--week-index", type=int, default=None, help="Target week index (default: last)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Checkpoints directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--explain", action="store_true", help="Print explanation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg)

    print(f"\n  Loading database: {args.database}")
    db_id = args.db_id or Path(args.database).stem
    events = load_json_events(args.database, database_id=db_id, timezone=getattr(cfg.project, "timezone", None))
    profile = profile_database(events, db_id)

    print(f"  Preparing data ({len(events):,} events)...")
    prepared = prepare_data(events, cfg, {db_id: profile})

    target_idx = args.week_index if args.week_index is not None else len(prepared.weeks) - 1
    print(f"  Target week: {target_idx} ({prepared.weeks[target_idx].week_start})")

    # Load models
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else cfg.project_root / "checkpoints"
    occ_feat_dim = occurrence_feature_dim()
    seq_dim = prepared.week_feature_dim
    hist_feat_dim = temporal_feature_dim()
    bins_per_day = prepared.bins_per_day

    occ_model = OccurrenceResidualModel(
        feature_dim=occ_feat_dim,
        sequence_dim=seq_dim,
        num_tasks=prepared.num_tasks,
        num_databases=prepared.num_databases,
        hidden_size=cfg.occurrence_model.hidden_size,
        num_layers=cfg.occurrence_model.num_layers,
        dropout=cfg.occurrence_model.dropout,
        delta_range=cfg.occurrence_model.delta_range,
    ).to(device)

    tmp_model = TemporalResidualModel(
        sequence_dim=seq_dim,
        history_feature_dim=hist_feat_dim,
        num_tasks=prepared.num_tasks,
        num_databases=prepared.num_databases,
        max_occurrences=prepared.max_count_cap,
        hidden_size=cfg.temporal_model.hidden_size,
        num_layers=cfg.temporal_model.num_layers,
        dropout=cfg.temporal_model.dropout,
        num_day_classes=7,
        num_time_classes=bins_per_day,
    ).to(device)

    # Load checkpoints if they exist
    occ_ckpt = ckpt_dir / "occurrence_model.pt"
    tmp_ckpt = ckpt_dir / "temporal_model.pt"
    if occ_ckpt.exists():
        ckpt = torch.load(occ_ckpt, map_location=device, weights_only=False)
        occ_model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded occurrence checkpoint (epoch {ckpt.get('best_epoch', '?')})")
    else:
        print(f"  WARNING: No occurrence checkpoint found at {occ_ckpt}")

    if tmp_ckpt.exists():
        ckpt = torch.load(tmp_ckpt, map_location=device, weights_only=False)
        tmp_model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded temporal checkpoint (epoch {ckpt.get('best_epoch', '?')})")
    else:
        print(f"  WARNING: No temporal checkpoint found at {tmp_ckpt}")

    # Predict
    print(f"\n  Running prediction pipeline...")
    result = predict_week(prepared, target_idx, occ_model, tmp_model, cfg, device)

    # Explain
    if args.explain:
        explanation = explain_prediction(result, prepared.bin_minutes)
        print(explanation)

    # Summary
    validation = result["validation"]
    schedule = result["schedule"]
    print(f"\n  Results:")
    print(f"    Events scheduled: {len(schedule)}")
    print(f"    Conflict-free: {validation['is_valid']}")
    print(f"    Same-device overlaps: {validation['overlap_same_device_count']}")

    # Save output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Make JSON-serializable
        output_data = {
            "week_index": target_idx,
            "week_start": str(prepared.weeks[target_idx].week_start),
            "schedule": result["schedule"],
            "template_metadata": result["template_metadata"],
            "final_counts": result["final_counts"],
            "occurrence_delta": result["occurrence_delta"],
            "validation": {k: v for k, v in validation.items() if k != "details"},
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"    Output saved to: {out_path}")

    print()


if __name__ == "__main__":
    main()
