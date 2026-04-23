#!/usr/bin/env python3
"""
CLI: train — global training of occurrence + temporal residual models.

Usage:
    python train.py --config configs/train_global.yaml
    python train.py --databases bases_datos/nexus_schedule_10years.json
    python train.py --max-epochs 5  (quick smoke test)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import load_config, resolve_device
from data.adapters.json_adapter import load_json_events
from data.preprocessing import prepare_data, build_split_indices
from data.profiling import profile_database, save_profile
from data.registry import DatabaseRegistry, build_registry_from_config
from data.datasets import OccurrenceDataset, TemporalDataset
from data.schema import PreparedData
from evaluation.benchmark_baselines import run_all_baselines
from features.occurrence_features import occurrence_feature_dim
from features.temporal_features import temporal_feature_dim
from models.occurrence_residual import OccurrenceResidualModel
from models.temporal_residual import TemporalResidualModel
from retrieval.template_retriever import TemplateRetriever
from training.engine import fit_model
from training.losses import OccurrenceResidualLoss, TemporalResidualLoss
from training.reporting import TrainingReporter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hybrid schedule predictor")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--databases", nargs="*", default=None, help="Database JSON files")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────
    cfg = load_config(args.config)
    seed = args.seed or cfg.project.seed
    set_seed(seed)

    device = resolve_device(cfg)
    print(f"\n{'='*60}")
    print(f"  Hybrid Multi-DB Schedule Predictor — Training")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Seed: {seed}")

    # ── Configure runtime ────────────────────────────────────────
    if device.type == "cuda":
        if cfg.runtime.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        if cfg.runtime.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ── Load databases ───────────────────────────────────────────
    if args.databases:
        registry = DatabaseRegistry()
        for db_path in args.databases:
            db_id = Path(db_path).stem
            registry.register(db_id, path=db_path, timezone=getattr(cfg.project, "timezone", None))
    else:
        registry = build_registry_from_config(cfg)

    if registry.num_databases == 0:
        print("  ERROR: No databases configured. Use --databases or config.data.databases")
        sys.exit(1)

    print(f"  Databases: {registry.database_ids}")

    # ── Profile each database ────────────────────────────────────
    print("\n  Profiling databases...")
    all_events = []
    profiles = {}
    for db_id in registry.database_ids:
        events = registry.load(db_id)
        all_events.extend(events)
        prof = profile_database(events, db_id)
        profiles[db_id] = prof
        print(
            f"    {db_id}: {prof.num_events:,} events, {prof.num_weeks} weeks, "
            f"{prof.num_task_types} tasks, {'single' if prof.is_single_device else 'multi'}-device"
        )
        save_profile(prof, cfg.project_root / "reports" / f"{db_id}_profile.json")

    # ── Prepare data ─────────────────────────────────────────────
    print("\n  Preparing data...")
    prepared = prepare_data(all_events, cfg, profiles)
    print(f"    Total weeks: {len(prepared.weeks)}")
    print(f"    Task types: {prepared.num_tasks} — {prepared.task_names}")
    print(f"    Databases: {prepared.num_databases} — {prepared.database_ids}")

    # ── Split ────────────────────────────────────────────────────
    window_weeks = cfg.data.window_weeks
    train_indices, val_indices = build_split_indices(prepared, cfg.data.train_ratio, window_weeks)
    print(f"    Train weeks: {len(train_indices)}, Val weeks: {len(val_indices)}")

    # ── Build retriever ──────────────────────────────────────────
    retriever = TemplateRetriever(prepared)

    # ── Build datasets ───────────────────────────────────────────
    print("\n  Building datasets...")
    occ_train_ds = OccurrenceDataset(prepared, train_indices, window_weeks, retriever, show_progress=True)
    occ_val_ds = OccurrenceDataset(prepared, val_indices, window_weeks, retriever)
    print(f"    Occurrence: train={len(occ_train_ds)}, val={len(occ_val_ds)}")

    tmp_train_ds = TemporalDataset(prepared, train_indices, window_weeks, retriever, show_progress=True)
    tmp_val_ds = TemporalDataset(prepared, val_indices, window_weeks, retriever)
    print(f"    Temporal:   train={len(tmp_train_ds)}, val={len(tmp_val_ds)}")

    # ── Create models ────────────────────────────────────────────
    occ_feat_dim = occurrence_feature_dim()
    seq_dim = prepared.week_feature_dim
    hist_feat_dim = temporal_feature_dim()
    bins_per_day = prepared.bins_per_day

    occ_cfg = cfg.occurrence_model
    occ_model = OccurrenceResidualModel(
        feature_dim=occ_feat_dim,
        sequence_dim=seq_dim,
        num_tasks=prepared.num_tasks,
        num_databases=prepared.num_databases,
        hidden_size=occ_cfg.hidden_size,
        num_layers=occ_cfg.num_layers,
        dropout=occ_cfg.dropout,
        task_embed_dim=occ_cfg.task_embed_dim,
        db_embed_dim=occ_cfg.db_embed_dim,
        delta_range=occ_cfg.delta_range,
    ).to(device)

    tmp_cfg = cfg.temporal_model
    max_occ = prepared.max_count_cap
    tmp_model = TemporalResidualModel(
        sequence_dim=seq_dim,
        history_feature_dim=hist_feat_dim,
        num_tasks=prepared.num_tasks,
        num_databases=prepared.num_databases,
        max_occurrences=max_occ,
        hidden_size=tmp_cfg.hidden_size,
        num_layers=tmp_cfg.num_layers,
        dropout=tmp_cfg.dropout,
        task_embed_dim=tmp_cfg.task_embed_dim,
        db_embed_dim=tmp_cfg.db_embed_dim,
        occ_embed_dim=tmp_cfg.occ_embed_dim,
        day_embed_dim=tmp_cfg.day_embed_dim,
        num_day_classes=7,
        num_time_classes=bins_per_day,
    ).to(device)

    occ_params = sum(p.numel() for p in occ_model.parameters())
    tmp_params = sum(p.numel() for p in tmp_model.parameters())
    print(f"\n  Occurrence model: {occ_params:,} parameters")
    print(f"  Temporal model:   {tmp_params:,} parameters")

    # ── DataLoaders ──────────────────────────────────────────────
    occ_bs = occ_cfg.batch_size
    tmp_bs = tmp_cfg.batch_size
    num_workers = cfg.runtime.num_workers

    occ_train_loader = DataLoader(occ_train_ds, batch_size=occ_bs, shuffle=True, num_workers=num_workers, pin_memory=cfg.runtime.pin_memory)
    occ_val_loader = DataLoader(occ_val_ds, batch_size=occ_bs, shuffle=False, num_workers=num_workers, pin_memory=cfg.runtime.pin_memory)
    tmp_train_loader = DataLoader(tmp_train_ds, batch_size=tmp_bs, shuffle=True, num_workers=num_workers, pin_memory=cfg.runtime.pin_memory)
    tmp_val_loader = DataLoader(tmp_val_ds, batch_size=tmp_bs, shuffle=False, num_workers=num_workers, pin_memory=cfg.runtime.pin_memory)

    # ── Reporters ────────────────────────────────────────────────
    reports_dir = cfg.project_root / getattr(cfg.reporting, "reports_dir", "reports")
    checkpoints_dir = cfg.project_root / getattr(cfg.reporting, "checkpoints_dir", "checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    occ_reporter = TrainingReporter(reports_dir / "occurrence")
    tmp_reporter = TrainingReporter(reports_dir / "temporal")

    # ── Train Occurrence Model ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Occurrence Residual Model")
    print(f"{'='*60}")

    occ_max_epochs = args.max_epochs or occ_cfg.max_epochs
    occ_optimizer = torch.optim.AdamW(occ_model.parameters(), lr=occ_cfg.lr, weight_decay=occ_cfg.weight_decay)
    occ_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(occ_optimizer, patience=occ_cfg.scheduler_patience, factor=0.5)
    occ_loss_fn = OccurrenceResidualLoss(change_weight=occ_cfg.change_loss_weight, delta_weight=occ_cfg.delta_loss_weight, delta_range=occ_cfg.delta_range)

    # Wrap model forward for the engine
    class OccModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, **kwargs):
            return self.model(kwargs["input_sequence"], kwargs["input_task_features"], kwargs["input_task_ids"], kwargs["input_db_ids"])

    occ_wrapper = OccModelWrapper(occ_model).to(device)

    def occ_loss_wrapper(outputs, batch):
        return occ_loss_fn(outputs, batch)

    occ_state = fit_model(
        occ_wrapper, occ_train_loader, occ_val_loader,
        occ_optimizer, occ_scheduler, occ_loss_wrapper, None,
        device, occ_max_epochs, occ_cfg.patience,
        model_name="Occurrence",
        monitor_name="val_loss",
        monitor_mode="min",
        amp_enabled=cfg.runtime.use_amp,
        amp_dtype=cfg.runtime.amp_dtype,
        verbose_every=getattr(cfg.reporting, "verbose_every", 5),
        callback=lambda epoch, entry: occ_reporter.log_epoch(entry),
    )

    # Save checkpoint
    torch.save({
        "model_state_dict": occ_model.state_dict(),
        "best_epoch": occ_state.best_epoch,
        "config": {"feature_dim": occ_feat_dim, "seq_dim": seq_dim, "num_tasks": prepared.num_tasks},
    }, checkpoints_dir / "occurrence_model.pt")

    # ── Train Temporal Model ─────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training Temporal Residual Model")
    print(f"{'='*60}")

    tmp_max_epochs = args.max_epochs or tmp_cfg.max_epochs
    tmp_optimizer = torch.optim.AdamW(tmp_model.parameters(), lr=tmp_cfg.lr, weight_decay=tmp_cfg.weight_decay)
    tmp_scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(tmp_optimizer, patience=tmp_cfg.scheduler_patience, factor=0.5)
    tmp_loss_fn = TemporalResidualLoss(
        day_weight=tmp_cfg.day_loss_weight,
        time_weight=tmp_cfg.time_loss_weight,
        duration_weight=tmp_cfg.duration_loss_weight,
        day_label_smoothing=tmp_cfg.day_label_smoothing,
        time_label_smoothing=tmp_cfg.time_label_smoothing,
    )

    class TmpModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, **kwargs):
            return self.model(
                kwargs["input_sequence"], kwargs["input_task_id"],
                kwargs["input_db_id"], kwargs["input_occurrence_slot"],
                kwargs["input_history_features"], kwargs["input_predicted_count_norm"],
                kwargs["input_occurrence_progress"], kwargs["input_anchor_day"],
                kwargs["input_anchor_time_bin"],
            )

    tmp_wrapper = TmpModelWrapper(tmp_model).to(device)

    def tmp_loss_wrapper(outputs, batch):
        return tmp_loss_fn(outputs, batch)

    tmp_state = fit_model(
        tmp_wrapper, tmp_train_loader, tmp_val_loader,
        tmp_optimizer, tmp_scheduler_lr, tmp_loss_wrapper, None,
        device, tmp_max_epochs, tmp_cfg.patience,
        model_name="Temporal",
        monitor_name="val_loss",
        monitor_mode="min",
        amp_enabled=cfg.runtime.use_amp,
        amp_dtype=cfg.runtime.amp_dtype,
        verbose_every=getattr(cfg.reporting, "verbose_every", 5),
        callback=lambda epoch, entry: tmp_reporter.log_epoch(entry),
    )

    # Save checkpoint
    torch.save({
        "model_state_dict": tmp_model.state_dict(),
        "best_epoch": tmp_state.best_epoch,
        "config": {"seq_dim": seq_dim, "hist_dim": hist_feat_dim, "num_tasks": prepared.num_tasks},
    }, checkpoints_dir / "temporal_model.pt")

    # ── Run baselines ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Running Baseline Comparisons")
    print(f"{'='*60}")

    baseline_results = run_all_baselines(
        prepared, val_indices, retriever, prepared.bin_minutes
    )
    for name, metrics in baseline_results.items():
        print(f"\n  {name}:")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"    {k}: {v:.2f}")

    # ── Save reports ─────────────────────────────────────────────
    occ_reporter.save_final_report(
        "OccurrenceResidual",
        occ_state.best_epoch,
        {"best_val_loss": occ_state.best_metric},
        baseline_results=baseline_results,
    )
    tmp_reporter.save_final_report(
        "TemporalResidual",
        tmp_state.best_epoch,
        {"best_val_loss": tmp_state.best_metric},
    )

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"{'='*60}")
    print(f"  Occurrence: best epoch {occ_state.best_epoch}, val_loss={occ_state.best_metric:.4f}")
    print(f"  Temporal:   best epoch {tmp_state.best_epoch}, val_loss={tmp_state.best_metric:.4f}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"  Reports:     {reports_dir}")
    print()


if __name__ == "__main__":
    main()
