from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hybrid_schedule.decoding import decode_week_with_constraints
from hybrid_schedule.training.datasets import UnifiedWeekSlotDataset


class UnifiedWeekPredictor:
    def __init__(self, context, config: dict, model, device: torch.device, feature_scaling: dict | None = None):
        self.context = context
        self.config = config
        self.model = model
        self.device = device
        self.bin_minutes = int(config['calendar']['bin_minutes'])
        self.bins_per_day = int(24 * 60 / self.bin_minutes)
        self.max_slots = int(config['calendar']['max_slot_prototypes'])
        self.feature_scaling = feature_scaling or {}
        self.model.eval()

    @torch.no_grad()
    def predict_database_robot(self, database_id: str, robot_id: str) -> dict:
        series = self.context.series[(database_id, robot_id)]
        target_week_idx = len(series.week_starts)
        ds = UnifiedWeekSlotDataset(
            self.context,
            indices=[(database_id, robot_id, target_week_idx)],
            window_weeks=int(self.config['calendar']['window_weeks']),
            max_slots=self.max_slots,
            max_slots_by_task=self.config['calendar'].get('max_slots_by_task'),
            bin_minutes=self.bin_minutes,
            history_stats=self.feature_scaling.get('history'),
            numeric_stats=self.feature_scaling.get('numeric'),
        )
        batch = ds[0]
        batch = {k: (v.unsqueeze(0).to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = self.model(
            history=batch['history'],
            task_ids=batch['task_ids'],
            database_ids=batch['database_ids'],
            robot_ids=batch['robot_ids'],
            slot_ids=batch['slot_ids'],
            anchor_days=batch['anchor_days'],
            anchor_times=batch['anchor_times'],
            numeric_features=batch['numeric_features'],
            query_mask=batch['query_mask'],
        )

        active_probs = torch.sigmoid(outputs['active_logits'][0]).cpu().numpy()
        day_probs = torch.softmax(outputs['day_logits'][0], dim=-1).cpu().numpy()
        time_probs = torch.softmax(outputs['time_logits'][0], dim=-1).cpu().numpy()
        pred_durations = torch.clamp(torch.round(torch.expm1(outputs['pred_log_duration'][0])), min=1.0).cpu().numpy()

        task_ids = batch['task_ids'][0].cpu().numpy()
        slot_ids = batch['slot_ids'][0].cpu().numpy()
        anchor_days = batch['anchor_days'][0].cpu().numpy()
        anchor_times = batch['anchor_times'][0].cpu().numpy()
        query_mask = batch['query_mask'][0].cpu().numpy().astype(bool)
        anchor_starts = anchor_days * self.bins_per_day + anchor_times

        selected = []
        threshold = float(self.config['inference'].get('active_threshold', 0.42))
        min_keep_prob = float(self.config['inference'].get('min_keep_prob', 0.20))
        per_task_caps = self.config['inference'].get('max_selected_slots_per_task_by_task')
        if per_task_caps is None:
            per_task_caps = self.config['calendar'].get('max_slots_by_task')
        if per_task_caps is None:
            per_task_caps = [int(self.config['inference'].get('max_selected_slots_per_task', self.max_slots))] * len(self.context.task_names)
        per_task_caps = [int(x) for x in per_task_caps]

        for task_idx in range(len(self.context.task_names)):
            indices = np.where((task_ids == task_idx) & query_mask)[0].tolist()
            if not indices:
                continue
            probs = np.asarray([active_probs[i] for i in indices], dtype=np.float32)
            task_cap = int(per_task_caps[task_idx])
            expected_count = int(np.clip(np.rint(probs.sum()), 0, min(len(indices), task_cap)))
            ranked = sorted(indices, key=lambda i: float(active_probs[i]), reverse=True)
            chosen = [i for i in ranked if float(active_probs[i]) >= threshold][:task_cap]
            if len(chosen) < expected_count:
                for i in ranked:
                    if i not in chosen and float(active_probs[i]) >= min_keep_prob:
                        chosen.append(i)
                    if len(chosen) >= expected_count:
                        break
            for i in chosen:
                selected.append({
                    'database_id': database_id,
                    'robot_id': robot_id,
                    'task_idx': int(task_ids[i]),
                    'task_type': self.context.task_names[int(task_ids[i])],
                    'slot_id': int(slot_ids[i]),
                    'active_prob': float(active_probs[i]),
                    'anchor_start_bin': int(anchor_starts[i]),
                    'anchor_duration_bins': int(max(1, int(pred_durations[i]))),
                    'pred_duration_bins': int(max(1, int(pred_durations[i]))),
                    'day_distribution': [(int(j), float(day_probs[i, j])) for j in np.argsort(day_probs[i])[::-1][: int(self.config['inference'].get('topk_days', 3))]],
                    'time_distribution': [(int(j), float(time_probs[i, j])) for j in np.argsort(time_probs[i])[::-1][: int(self.config['inference'].get('topk_times', 12))]],
                })

        decoded = decode_week_with_constraints(
            selected_events=selected,
            bins_per_day=self.bins_per_day,
            topk_days=int(self.config['inference'].get('topk_days', 3)),
            topk_times=int(self.config['inference'].get('topk_times', 12)),
            duration_radius_bins=int(self.config['inference'].get('duration_radius_bins', 1)),
            beam_width=int(self.config['inference'].get('beam_width', 6)),
            max_candidates_per_event=int(self.config['inference'].get('max_candidates_per_event', 18)),
            anchor_penalty=float(self.config['inference'].get('anchor_penalty', 0.10)),
            duration_penalty=float(self.config['inference'].get('duration_penalty', 0.05)),
            occupancy_soft_penalty=float(self.config['inference'].get('occupancy_soft_penalty', 5.0)),
        )

        week_start = pd.Timestamp(series.week_starts[-1]) + (pd.Timestamp(series.week_starts[-1]) - pd.Timestamp(series.week_starts[-2]) if len(series.week_starts) >= 2 else pd.Timedelta(days=7))
        final_rows = []
        for evt in decoded:
            start_ts = week_start + pd.Timedelta(minutes=int(evt['start_bin']) * self.bin_minutes)
            end_ts = start_ts + pd.Timedelta(minutes=int(evt['duration_bins']) * self.bin_minutes)
            final_rows.append({
                **evt,
                'start_time': str(start_ts),
                'end_time': str(end_ts),
                'duration_minutes': int(evt['duration_bins']) * self.bin_minutes,
            })

        return {
            'database_id': database_id,
            'robot_id': robot_id,
            'week_start': str(week_start),
            'bin_minutes': self.bin_minutes,
            'events': final_rows,
        }

    def save_prediction(self, database_id: str, robot_id: str, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.predict_database_robot(database_id, robot_id)
        out_path = output_dir / f'prediction_{database_id}_{robot_id}.json'
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return out_path
