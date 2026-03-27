import torch
from torch.utils.data import DataLoader, Subset

from dataset_3 import WeekDataset
from sec2sec.modelo_sec2sec import WeekPredictor
from sec2sec.loss import compute_loss
from sec2sec.matric import compute_metrics
from dataset import semanas_separadas, df_training

dataset = WeekDataset(semanas_separadas)

# ── Split cronológico ─────────────────────────────────────────────────────────
# Las primeras 80% de ventanas (más antiguas) → entrenamiento
# Las últimas  20% de ventanas (más recientes) → validación
# Así la validación evalúa semanas que el modelo nunca ha visto,
# igual que en producción real.
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

train_set = Subset(dataset, range(train_size))
val_set   = Subset(dataset, range(train_size, len(dataset)))

print(f"Muestras de entrenamiento : {train_size}")
print(f"Muestras de validación    : {val_size}")
print(f"  → Train cubre las primeras {train_size} ventanas (pasado)")
print(f"  → Val   cubre las últimas  {val_size}  ventanas (futuro)\n")

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=4, shuffle=False)  # sin shuffle en val

num_tasks = df_training['task_id'].nunique()
model = WeekPredictor(num_tasks=num_tasks, embed_dim=16, hidden_size=64, num_layers=1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=20, factor=0.5
)

# ── Early stopping ────────────────────────────────────────────────────────────
PATIENCE          = 80
MIN_DELTA         = 1e-4
best_val_loss     = float('inf')
best_epoch        = 0
epochs_no_improve = 0
best_state        = None
# ─────────────────────────────────────────────────────────────────────────────

for epoch in range(1000):
    # ── TRAIN ─────────────────────────────────────────────────────────────────
    model.train()
    train_loss    = 0.0
    train_metrics = {'acc_task': 0, 'acc_day': 0, 'acc_hour': 0,
                     'acc_minute_bin': 0, 'acc_minute_±5m': 0, 'mae_duration': 0}

    # Teacher forcing: empieza en 1.0 y baja a 0.0 a lo largo de 500 épocas
    tf_ratio = max(0.0, 1.0 - epoch / 500)

    for X_batch, y_batch, mask_X, mask_y in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch, masks_X=mask_X, target=y_batch,
                      teacher_forcing_ratio=tf_ratio)
        loss  = compute_loss(preds, y_batch, mask_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()
        for k, v in compute_metrics(preds, y_batch, mask_y).items():
            train_metrics[k] += v

    n = len(train_loader)
    train_loss /= n
    for k in train_metrics:
        train_metrics[k] /= n

    # ── VALIDACIÓN ────────────────────────────────────────────────────────────
    # Sin teacher forcing: el modelo usa sus propias predicciones,
    # igual que en producción.
    model.eval()
    val_loss    = 0.0
    val_metrics = {k: 0 for k in train_metrics}

    with torch.no_grad():
        for X_batch, y_batch, mask_X, mask_y in val_loader:
            preds = model(X_batch, masks_X=mask_X, target=None,
                          teacher_forcing_ratio=0.0)   # ← sin teacher forcing
            val_loss += compute_loss(preds, y_batch, mask_y).item()
            for k, v in compute_metrics(preds, y_batch, mask_y).items():
                val_metrics[k] += v

    n = len(val_loader)
    val_loss /= n
    for k in val_metrics:
        val_metrics[k] /= n

    scheduler.step(val_loss)

    # ── Early stopping ────────────────────────────────────────────────────────
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss     = val_loss
        best_epoch        = epoch
        epochs_no_improve = 0
        best_state        = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"\n⏹  Early stopping en época {epoch} "
              f"(sin mejora desde época {best_epoch}, best val_loss={best_val_loss:.4f})")
        model.load_state_dict(best_state)
        break
    # ─────────────────────────────────────────────────────────────────────────

    if epoch % 10 == 0:
        marker = " ✓" if epochs_no_improve == 0 else f" (sin mejora: {epochs_no_improve}/{PATIENCE})"
        print(f"\n{'='*65}")
        print(f"Epoch {epoch:3d}{marker}")
        print(f"{'─'*65}")
        print(f"{'Métrica':<25} {'Train':>10} {'Val':>10}")
        print(f"{'─'*65}")
        print(f"{'Loss':<25} {train_loss:>9.4f}  {val_loss:>9.4f}")
        print(f"{'Acc Tarea':<25} {train_metrics['acc_task']:>8.1f}%  {val_metrics['acc_task']:>8.1f}%")
        print(f"{'Acc Día':<25} {train_metrics['acc_day']:>8.1f}%  {val_metrics['acc_day']:>8.1f}%")
        print(f"{'Acc Hora':<25} {train_metrics['acc_hour']:>8.1f}%  {val_metrics['acc_hour']:>8.1f}%")
        print(f"{'Acc Minuto (bin exacto)':<25} {train_metrics['acc_minute_bin']:>8.1f}%  {val_metrics['acc_minute_bin']:>8.1f}%")
        print(f"{'Acc Minuto (±5 min)':<25} {train_metrics['acc_minute_±5m']:>8.1f}%  {val_metrics['acc_minute_±5m']:>8.1f}%")
        print(f"{'MAE Duración':<25} {train_metrics['mae_duration']:>9.4f}  {val_metrics['mae_duration']:>9.4f}")
        print(f"{'='*65}")

print(f"\n✅ Entrenamiento finalizado. Mejor época: {best_epoch} | Best val_loss: {best_val_loss:.4f}")