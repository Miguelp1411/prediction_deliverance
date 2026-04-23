"""
slot_train.py
─────────────
Script de entrenamiento para el RoutinePredictor.

Mejoras respecto al entrenamiento anterior:
  - Sin teacher forcing (predicción paralela → no necesario)
  - CosineAnnealingWarmRestarts: escapa mínimos locales
  - Guardado del mejor modelo con sus metadatos
  - Baseline estadístico al inicio para referencia
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from proyectos_anteriores.previo.slot.slot_dataset import SlotWeekDataset
from proyectos_anteriores.previo.slot.slot_model   import RoutinePredictor, model_summary
from proyectos_anteriores.previo.slot.slot_loss    import compute_loss, compute_metrics

# ── Importar datos (misma pipeline que antes) ─────────────────────────────────
from proyectos_anteriores.previo.dataset import semanas_separadas, df_training

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = SlotWeekDataset(semanas_separadas, min_slot_freq=2)

# Split cronológico 80/20
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

train_set = Subset(dataset, range(train_size))
val_set   = Subset(dataset, range(train_size, len(dataset)))

print(f"Train: {train_size} ventanas | Val: {val_size} ventanas")

# Batch size mayor que antes (slots son independientes entre sí)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

# ── Baseline estadístico (referencia) ─────────────────────────────────────────
print("\n📊 Baseline estadístico (predecir siempre la frecuencia histórica):")
slot_freq = torch.tensor(dataset.slot_freq)

# Simulamos el dataset de validación con el baseline
val_data_all = [dataset[i] for i in range(train_size, len(dataset))]
if val_data_all:
    y_all = torch.stack([v[1] for v in val_data_all])    # (N_val, K, 4)
    tgt_occurs = y_all[:, :, 0] > 0.5

    # Baseline: predecir slot como activo si su frecuencia histórica > 0.5
    pred_baseline = slot_freq.unsqueeze(0).expand_as(tgt_occurs) > 0.5
    tp = (pred_baseline & tgt_occurs).float().sum()
    fp = (pred_baseline & ~tgt_occurs).float().sum()
    fn = (~pred_baseline & tgt_occurs).float().sum()
    prec = (tp / (tp + fp + 1e-8)).item()
    rec  = (tp / (tp + fn + 1e-8)).item()
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    print(f"  F1 Slot   : {f1*100:.1f}%  |  Precision: {prec*100:.1f}%  |  Recall: {rec*100:.1f}%")
    print(f"  → El modelo debe superar esto para aportar valor.\n")

# ── Modelo ────────────────────────────────────────────────────────────────────
model = RoutinePredictor(
    num_slots = dataset.num_slots,
    d_model   = 64,
    n_heads   = 4,
    n_layers  = 2,
    dropout   = 0.3,
    window    = 4,
    input_dim = 6,
)
model_summary(model, dataset.num_slots)

# Frecuencia de slots para la loss (pasamos al compute_loss)
slot_freq_tensor = torch.tensor(dataset.slot_freq)

# ── Optimizador ───────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = 1e-3,
    weight_decay = 1e-2,   # L2 más fuerte que antes (era 1e-3)
)

# CosineAnnealingWarmRestarts: cada T_0=100 épocas reinicia el LR
# Ayuda a escapar mínimos locales — especialmente útil con pocos datos
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=100, T_mult=2, eta_min=1e-5
)

# ── Early stopping ────────────────────────────────────────────────────────────
PATIENCE      = 100
MIN_DELTA     = 1e-4
best_val_loss = float('inf')
best_epoch    = 0
no_improve    = 0
best_state    = None

# ── Loop de entrenamiento ─────────────────────────────────────────────────────
for epoch in range(600):

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    train_m    = {k: 0.0 for k in ['f1_tarea', 'f1_slot', 'precision_slot', 'recall_slot',
                                     'acc_hour', 'acc_min_exact', 'acc_min_5', 'mae_dur']}

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = compute_loss(preds, y_batch, slot_freq_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
        for k, v in compute_metrics(preds, y_batch, slots=dataset.slots).items():
            train_m[k] += v

    scheduler.step()

    n = len(train_loader)
    train_loss /= n
    for k in train_m:
        train_m[k] /= n

    # ── VAL ───────────────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_m    = {k: 0.0 for k in train_m}

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch)
            val_loss += compute_loss(preds, y_batch, slot_freq_tensor).item()
            for k, v in compute_metrics(preds, y_batch, slots=dataset.slots).items():
                val_m[k] += v

    n = len(val_loader)
    val_loss /= n
    for k in val_m:
        val_m[k] /= n

    # ── Early stopping ────────────────────────────────────────────────────────
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        best_epoch    = epoch
        no_improve    = 0
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        no_improve += 1

    if no_improve >= PATIENCE:
        print(f"\n⏹  Early stopping en época {epoch} "
              f"(sin mejora desde época {best_epoch})")
        model.load_state_dict(best_state)
        break

    # ── Log cada 10 épocas ────────────────────────────────────────────────────
    if epoch % 10 == 0:
        marker = " ✓" if no_improve == 0 else f" (sin mejora: {no_improve}/{PATIENCE})"
        lr_now = optimizer.param_groups[0]['lr']

        print(f"\n{'='*72}")
        print(f"Época {epoch:4d}{marker}  |  LR: {lr_now:.2e}")
        print(f"{'─'*72}")
        print(f"{'Métrica':<36} {'Train':>8} {'Val':>8}")
        print(f"{'─'*72}")
        print(f"{'Loss':<36} {train_loss:>7.4f}  {val_loss:>7.4f}")
        print(f"{'─'*72}")
        print(f"  ① TAREA  (¿qué tareas hay esta semana?)")
        print(f"{'  F1 Tarea':<36} {train_m['f1_tarea']:>7.1f}%  {val_m['f1_tarea']:>7.1f}%")
        print(f"{'─'*72}")
        print(f"  ② SLOT   (¿tarea correcta + día correcto?)")
        print(f"{'  F1 Slot':<36} {train_m['f1_slot']:>7.1f}%  {val_m['f1_slot']:>7.1f}%")
        print(f"{'  Precision Slot':<36} {train_m['precision_slot']:>7.1f}%  {val_m['precision_slot']:>7.1f}%")
        print(f"{'  Recall Slot':<36} {train_m['recall_slot']:>7.1f}%  {val_m['recall_slot']:>7.1f}%")
        print(f"{'─'*72}")
        print(f"  ③ TIMING (hora, minuto, duración)")
        print(f"{'  Acc Hora':<36} {train_m['acc_hour']:>7.1f}%  {val_m['acc_hour']:>7.1f}%")
        print(f"{'  Acc Minuto exacto':<36} {train_m['acc_min_exact']:>7.1f}%  {val_m['acc_min_exact']:>7.1f}%")
        print(f"{'  Acc Minuto ±5 min':<36} {train_m['acc_min_5']:>7.1f}%  {val_m['acc_min_5']:>7.1f}%")
        print(f"{'  MAE Duración':<36} {train_m['mae_dur']:>8.4f}  {val_m['mae_dur']:>8.4f}")
        print(f"{'='*72}")

print(f"\n✅ Mejor época: {best_epoch} | Best val_loss: {best_val_loss:.4f}")

# ── Guardar modelo ────────────────────────────────────────────────────────────
torch.save({
    'model_state'  : model.state_dict(),
    'model_config' : {
        'num_slots': dataset.num_slots,
        'd_model'  : 64,
        'n_heads'  : 4,
        'n_layers' : 2,
        'dropout'  : 0.3,
        'window'   : 4,
        'input_dim': 6,
    },
    'slots'        : dataset.slots,
    'slot_to_idx'  : dataset.slot_to_idx,
    'slot_freq'    : dataset.slot_freq,
    'dur_mean'     : dataset.dur_mean,
    'dur_std'      : dataset.dur_std,
    'best_epoch'   : best_epoch,
    'best_val_loss': best_val_loss,
}, 'routine_predictor.pt')

print("💾 Modelo guardado en routine_predictor.pt")