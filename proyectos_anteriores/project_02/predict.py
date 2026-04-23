from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from proyectos_anteriores.project_02.config import (
    BIN_MINUTES,
    CHECKPOINT_DIR,
    DATA_PATH,
    DEVICE,
    TIMEZONE,
    WINDOW_WEEKS,
)
from proyectos_anteriores.project_02.data.io import load_tasks_dataframe
from proyectos_anteriores.project_02.data.preprocessing import (
    build_history_features,
    denormalize_duration,
    prepare_data,
    week_to_feature_vector,
)
from proyectos_anteriores.project_02.models.occurrence_model import TaskOccurrenceModel
from proyectos_anteriores.project_02.models.temporal_model import TemporalAssignmentModel
from proyectos_anteriores.project_02.utils.serialization import load_checkpoint


def predict_next_week(
    occurrence_model,
    temporal_model,
    prepared,
    target_week_idx,
    device,
):
    """
    Genera la predicción completa de una semana.
    """

    occurrence_model.eval()
    temporal_model.eval()

    with torch.no_grad():
        # Construimos la secuencia de semanas de contexto (ventana)
        context_weeks = prepared.weeks[max(0, target_week_idx - WINDOW_WEEKS): target_week_idx]
        if not context_weeks:
            # Fallback si no hay historia suficiente
            seq = np.zeros((WINDOW_WEEKS, prepared.week_feature_dim), dtype=np.float32)
        else:
            seq = np.stack([week_to_feature_vector(week) for week in context_weeks]).astype(np.float32)
            # Pad si hubiese menos semanas
            if seq.shape[0] < WINDOW_WEEKS:
                pad = np.zeros((WINDOW_WEEKS - seq.shape[0], prepared.week_feature_dim), dtype=np.float32)
                seq = np.concatenate([pad, seq], axis=0)

        sequence_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        # =========================
        # 1) Predicción de conteos
        # =========================
        count_logits = occurrence_model(sequence_tensor)
        pred_counts = torch.argmax(count_logits, dim=-1).cpu().numpy()[0]

        predictions = []

        # =========================
        # 2) Predicción temporal
        # =========================
        for task_id, count in enumerate(pred_counts):
            if count == 0:
                continue
                
            history_features_np = build_history_features(
                weeks=prepared.weeks,
                target_week_index=target_week_idx,
                task_id=task_id,
                duration_min=prepared.duration_min,
                duration_max=prepared.duration_max,
            )
            history_tensor = torch.tensor(history_features_np, dtype=torch.float32).unsqueeze(0).to(device)
            predicted_count_norm_tensor = torch.tensor(
                [float(count / prepared.max_count_cap)], dtype=torch.float32
            ).to(device)

            for occ_idx in range(count):
                task_tensor = torch.tensor([task_id]).to(device)
                occ_tensor = torch.tensor([occ_idx]).to(device)

                start_logits, pred_duration_norm = temporal_model(
                    sequence=sequence_tensor,
                    task_id=task_tensor,
                    occurrence_index=occ_tensor,
                    history_features=history_tensor,
                    predicted_count_norm=predicted_count_norm_tensor,
                )

                start_bin = torch.argmax(start_logits, dim=-1).item()
                duration = denormalize_duration(
                    pred_duration_norm.item(),
                    prepared.duration_min,
                    prepared.duration_max,
                )

                predictions.append(
                    {
                        "task_name": prepared.task_names[task_id],
                        "start_bin": int(start_bin),
                        "duration": float(duration),
                    }
                )

    return predictions


def _resolve_device(device_override: str | None) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_iso_z(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        return ts.isoformat()
    ts_utc = ts.tz_convert("UTC")
    return ts_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _load_models(occ_path: Path, tmp_path: Path, device: torch.device):
    occ_payload = load_checkpoint(occ_path, map_location=device)
    tmp_payload = load_checkpoint(tmp_path, map_location=device)

    occ_hparams = occ_payload["model_hparams"]
    tmp_hparams = tmp_payload["model_hparams"]

    occ_model = TaskOccurrenceModel(**occ_hparams).to(device)
    occ_model.load_state_dict(occ_payload["state_dict"])

    tmp_model = TemporalAssignmentModel(**tmp_hparams).to(device)
    tmp_model.load_state_dict(tmp_payload["state_dict"])

    occ_model.eval()
    tmp_model.eval()

    return occ_model, tmp_model, occ_payload.get("metadata", {})


def _validate_prepared(prepared, metadata: dict):
    if not metadata:
        return

    if metadata.get("task_names") and prepared.task_names != metadata["task_names"]:
        raise ValueError(
            "Las tareas del JSON no coinciden con las del modelo entrenado. "
            "Asegúrate de usar datos compatibles con los checkpoints."
        )

    meta_max = metadata.get("max_count_cap")
    if meta_max is not None and prepared.max_count_cap != meta_max:
        raise ValueError(
            f"max_count_cap no coincide (data={prepared.max_count_cap}, model={meta_max}). "
            "Entrena el modelo con la misma base de datos."
        )

    if metadata.get("week_feature_dim") and prepared.week_feature_dim != metadata["week_feature_dim"]:
        raise ValueError("week_feature_dim no coincide con el modelo entrenado.")

    if metadata.get("history_feature_dim") and prepared.history_feature_dim != metadata["history_feature_dim"]:
        raise ValueError("history_feature_dim no coincide con el modelo entrenado.")


def _build_output(predictions: list[dict], week_start: pd.Timestamp) -> list[dict]:
    output = []
    for item in predictions:
        start_time = week_start + pd.Timedelta(minutes=int(item["start_bin"]) * BIN_MINUTES)
        end_time = start_time + pd.Timedelta(minutes=float(item["duration"]))
        output.append(
            {
                "task_name": item["task_name"],
                "start_time": _to_iso_z(start_time),
                "end_time": _to_iso_z(end_time),
                "start_bin": int(item["start_bin"]),
                "duration_minutes": float(item["duration"]),
            }
        )

    output.sort(key=lambda x: (x["start_time"], x["task_name"]))
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Genera la próxima semana de tareas desde otra base de datos y guarda un JSON."
    )
    parser.add_argument("--data", default=str(DATA_PATH), help="Ruta al JSON con tareas de entrada.")
    parser.add_argument(
        "--output",
        default=str(CHECKPOINT_DIR / "predicted_next_week.json"),
        help="Ruta de salida del JSON con la semana generada.",
    )
    parser.add_argument(
        "--occ-checkpoint",
        default=str(CHECKPOINT_DIR / "occurrence_model.pt"),
        help="Checkpoint del modelo de ocurrencias.",
    )
    parser.add_argument(
        "--tmp-checkpoint",
        default=str(CHECKPOINT_DIR / "temporal_model.pt"),
        help="Checkpoint del modelo temporal.",
    )
    parser.add_argument(
        "--timezone",
        default=TIMEZONE,
        help="Zona horaria a la que convertir los timestamps (ej: Europe/Madrid).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override del dispositivo (cpu o cuda).",
    )

    args = parser.parse_args()

    device = _resolve_device(args.device)
    df = load_tasks_dataframe(args.data, timezone=args.timezone)
    prepared = prepare_data(df, train_ratio=1.0)

    occ_model, tmp_model, metadata = _load_models(
        Path(args.occ_checkpoint),
        Path(args.tmp_checkpoint),
        device,
    )
    _validate_prepared(prepared, metadata)

    target_week_idx = len(prepared.weeks)
    if not prepared.weeks:
        raise ValueError("No hay semanas históricas en la base de datos de entrada.")
    target_week_start = prepared.weeks[-1].week_start + pd.Timedelta(days=7)

    predictions = predict_next_week(
        occurrence_model=occ_model,
        temporal_model=tmp_model,
        prepared=prepared,
        target_week_idx=target_week_idx,
        device=device,
    )

    output = _build_output(predictions, target_week_start)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Semana generada: {out_path}")
    print(f"Tareas predichas: {len(output)}")


if __name__ == "__main__":
    main()
