"""
slot_predict.py
───────────────
Genera la predicción de una semana completa usando el RoutinePredictor.

Uso:
    from slot_predict import predict_next_week

    # context_weeks: lista de 4 DataFrames (las 4 semanas anteriores)
    df_pred = predict_next_week(model, dataset, context_weeks, threshold=0.4)
    print(df_pred[['task_name', 'start_time', 'end_time']])
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from proyectos_anteriores.previo.slot.slot_dataset import SlotWeekDataset
from proyectos_anteriores.previo.slot.slot_model   import RoutinePredictor


def predict_next_week(
    model       : RoutinePredictor,
    dataset     : SlotWeekDataset,
    context_weeks,              # lista de 4 DataFrames con las semanas previas
    label_encoder,              # sklearn LabelEncoder usado en dataset.py
    threshold   : float = 0.4, # umbral de confianza para activar un slot
    target_monday = None,       # datetime del lunes de la semana a predecir
) -> pd.DataFrame:
    """
    Genera predicciones para la próxima semana en formato JSON-friendly.

    threshold: bajar para recordar más tareas (más recall), subir para mayor precisión.
    """
    model.eval()

    # ── Codificar contexto ────────────────────────────────────────────────────
    ctx = np.stack([
        dataset._encode_week(w, recency=(j + 1) / 4)
        for j, w in enumerate(context_weeks)
    ])                                                          # (WINDOW, K, 6)
    ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)   # (1, WINDOW, K, 6)

    # ── Inferencia ────────────────────────────────────────────────────────────
    with torch.no_grad():
        occurs, hour, minute, duration = model(ctx_tensor)

    prob_occurs  = torch.sigmoid(occurs)[0]    # (K,)
    pred_hour    = hour[0].argmax(dim=-1)      # (K,)
    pred_min_bin = minute[0].argmax(dim=-1)    # (K,)
    pred_dur     = duration[0]                 # (K,)

    # ── Calcular fecha base (lunes de la semana a predecir) ───────────────────
    if target_monday is None:
        last_date    = context_weeks[-1]['start_time'].max()
        days_to_mon  = (7 - last_date.weekday()) % 7
        days_to_mon  = days_to_mon if days_to_mon > 0 else 7
        target_monday = (last_date + timedelta(days=days_to_mon)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    # ── Construir predicciones ────────────────────────────────────────────────
    predictions = []

    for slot_idx, (task_id, day_of_week) in enumerate(dataset.slots):
        confidence = prob_occurs[slot_idx].item()
        if confidence < threshold:
            continue

        hour_val    = int(pred_hour[slot_idx].item())
        minute_val  = int(pred_min_bin[slot_idx].item()) * 5        # bin → minuto real
        dur_norm    = float(pred_dur[slot_idx].item())
        dur_mins    = max(1.0, dur_norm * dataset.dur_std + dataset.dur_mean)

        # Fecha y hora de inicio
        task_date  = target_monday + timedelta(days=int(day_of_week))
        start_time = task_date.replace(hour=hour_val, minute=minute_val,
                                        second=0, microsecond=0)
        end_time   = start_time + timedelta(minutes=dur_mins)

        task_name = label_encoder.inverse_transform([task_id])[0]

        predictions.append({
            'task_name'  : task_name,
            'start_time' : start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'end_time'   : end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            'confidence' : round(confidence, 3),
            'day_of_week': int(day_of_week),
        })

    df = pd.DataFrame(predictions)
    if len(df) > 0:
        df = df.sort_values(['day_of_week', 'start_time']).reset_index(drop=True)

    return df


# ── Demo de uso ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from proyectos_anteriores.previo.dataset import semanas_separadas, label_encoder

    # Cargar modelo
    ckpt = torch.load('routine_predictor.pt', map_location='cpu')

    model = RoutinePredictor(**ckpt['model_config'])
    model.load_state_dict(ckpt['model_state'])

    # Reconstruir dataset con los mismos parámetros
    dataset = SlotWeekDataset(semanas_separadas, min_slot_freq=2)
    # Restaurar metadatos del checkpoint
    dataset.dur_mean = ckpt['dur_mean']
    dataset.dur_std  = ckpt['dur_std']
    dataset.slots    = ckpt['slots']

    # Predecir con las últimas 4 semanas disponibles
    context = semanas_separadas[-4:]
    df_pred = predict_next_week(model, dataset, context, label_encoder, threshold=0.4)

    print("\n📅 Predicción de la próxima semana:")
    print("=" * 60)
    days = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    for day_idx in range(7):
        day_tasks = df_pred[df_pred['day_of_week'] == day_index] if 'day_of_week' in df_pred.columns else pd.DataFrame()
        day_tasks = df_pred[df_pred['day_of_week'] == day_index] if False else \
                    df_pred[df_pred['start_time'].str.contains(f'T{str(day_index).zfill(2)}', na=False)]
        # Simplificado:
        pass

    for _, row in df_pred.iterrows():
        day_name = days[row['day_of_week']]
        print(f"  {day_name}  {row['start_time'][11:16]}–{row['end_time'][11:16]}"
              f"  {row['task_name']:<35}  ({row['confidence']:.0%})")

    print(f"\nTotal tareas predichas: {len(df_pred)}")
