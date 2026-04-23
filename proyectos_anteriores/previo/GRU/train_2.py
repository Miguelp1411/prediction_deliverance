import torch
import torch.nn as nn
import torch.optim as optim

from proyectos_anteriores.previo.dataset_2 import (
    label_encoder,
    semanas_train,
    semanas_val,
    semana_a_tensores,
    semana_a_ventanas,
    COLS_CONTINUAS,
)
from proyectos_anteriores.previo.GRU.modelo_gru_2 import GRU, top_k_accuracy

# ─────────────────────────────────────────────────────────────────────────────
# 1. HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────
NUM_TAREAS  = len(label_encoder.classes_)

EPOCAS      = 100       # alto porque early stopping parará antes
ALPHA       = 0.7       # peso clasificación vs regresión
TOP_K       = 3         # ventana para top-k accuracy
SEQ_LEN     = 10        # longitud de cada ventana deslizante

LR_INICIAL  = 0.005
STEP_SIZE   = 12        # era 8 → más suave para no matar el lr demasiado pronto
GAMMA       = 0.5       # factor de reducción del lr cada STEP_SIZE épocas

PACIENCIA   = 8         # épocas sin mejora en val_loss antes de parar (early stopping)

# ─────────────────────────────────────────────────────────────────────────────
# 2. MODELO, LOSS Y OPTIMIZADOR
# ─────────────────────────────────────────────────────────────────────────────
modelo = GRU(
    num_tareas_unicas       = NUM_TAREAS,
    embedding_dim           = 16,    # reducido respecto a v2
    hidden_size             = 32,    # reducido respecto a v2
    num_layers              = 1,     # reducido respecto a v2
    num_continuous_features = len(COLS_CONTINUAS),
    dropout                 = 0.4,   # aumentado respecto a v2
)

criterio_clasificacion = nn.CrossEntropyLoss()
criterio_regresion     = nn.MSELoss()

optimizador = optim.Adam(modelo.parameters(), lr=LR_INICIAL)

# Scheduler más suave: divide el LR cada STEP_SIZE épocas
scheduler = optim.lr_scheduler.StepLR(
    optimizador, step_size=STEP_SIZE, gamma=GAMMA
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PRE-GENERAR VENTANAS DE ENTRENAMIENTO
#
#    Hacemos esto una sola vez antes del bucle para no recalcular cada época.
#    semana_a_ventanas() convierte cada semana en N ventanas de longitud SEQ_LEN,
#    multiplicando el tamaño efectivo del dataset de entrenamiento.
# ─────────────────────────────────────────────────────────────────────────────
ventanas_train = []
for semana_df in semanas_train:
    ventanas_train.extend(semana_a_ventanas(semana_df, seq_len=SEQ_LEN))

print(f"Ventanas de entrenamiento generadas: {len(ventanas_train)}")
print(f"  (antes: {len(semanas_train)} muestras → ahora: {len(ventanas_train)} ventanas)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FUNCIÓN DE EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────
def evaluar(semanas):
    """
    Evalúa sobre secuencias completas (sin ventanas) para medir
    rendimiento en condiciones reales de predicción.
    """
    modelo.eval()
    loss_total = 0.0
    acc_total  = 0.0
    n          = 0

    with torch.no_grad():
        for semana_df in semanas:
            X_cat, X_cont, Y_cat, Y_cont = semana_a_tensores(semana_df)

            pred_cat, pred_cont, _ = modelo(X_cat, X_cont)

            pred_cat_flat = pred_cat.view(-1, NUM_TAREAS)
            Y_cat_flat    = Y_cat.view(-1)

            loss_cat  = criterio_clasificacion(pred_cat_flat, Y_cat_flat)
            loss_cont = criterio_regresion(pred_cont, Y_cont)
            loss      = ALPHA * loss_cat + (1 - ALPHA) * loss_cont

            loss_total += loss.item()
            acc_total  += top_k_accuracy(pred_cat_flat, Y_cat_flat, k=TOP_K)
            n          += 1

    modelo.train()
    return (loss_total / n, acc_total / n) if n > 0 else (0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUCLE DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  Tareas únicas  : {NUM_TAREAS}")
print(f"  Ventanas train : {len(ventanas_train)}  |  Semanas val: {len(semanas_val)}")
print(f"  Épocas máx.    : {EPOCAS}  |  Early stopping: paciencia={PACIENCIA}")
print(f"  LR inicial     : {LR_INICIAL}  |  Step={STEP_SIZE}  Gamma={GAMMA}")
print(f"  Alpha (cls)    : {ALPHA}  |  Top-k: {TOP_K}  |  Seq_len: {SEQ_LEN}")
print("=" * 70)

mejor_val_loss    = float('inf')
epocas_sin_mejora = 0

for epoca in range(1, EPOCAS + 1):

    # ── Entrenamiento ─────────────────────────────────────────────────────
    modelo.train()
    loss_train_total = 0.0
    acc_train_total  = 0.0

    for (X_cat, X_cont, Y_cat, Y_cont) in ventanas_train:

        optimizador.zero_grad()

        pred_cat, pred_cont, _ = modelo(X_cat, X_cont)

        pred_cat_flat = pred_cat.view(-1, NUM_TAREAS)
        Y_cat_flat    = Y_cat.view(-1)

        loss_cat  = criterio_clasificacion(pred_cat_flat, Y_cat_flat)
        loss_cont = criterio_regresion(pred_cont, Y_cont)
        loss      = ALPHA * loss_cat + (1 - ALPHA) * loss_cont

        loss.backward()
        optimizador.step()

        loss_train_total += loss.item()
        acc_train_total  += top_k_accuracy(pred_cat_flat, Y_cat_flat, k=TOP_K)

    scheduler.step()

    n_vent      = len(ventanas_train)
    train_loss  = loss_train_total / n_vent
    train_acc   = acc_train_total  / n_vent
    lr_actual   = scheduler.get_last_lr()[0]

    # ── Validación ───────────────────────────────────────────────────────
    val_loss, val_acc = evaluar(semanas_val)

    print(
        f"Época {epoca:>3}/{EPOCAS} | "
        f"Train {train_loss:.4f} acc@{TOP_K}: {train_acc:.2%} | "
        f"Val {val_loss:.4f} acc@{TOP_K}: {val_acc:.2%} | "
        f"LR: {lr_actual:.5f}"
    )

    # ── Early stopping ────────────────────────────────────────────────────
    if val_loss < mejor_val_loss:
        mejor_val_loss    = val_loss
        epocas_sin_mejora = 0
        torch.save(modelo.state_dict(), 'mejor_modelo.pt')
        print(f"           ↑ Nuevo mejor modelo guardado (val_loss={mejor_val_loss:.4f})")
    else:
        epocas_sin_mejora += 1
        if epocas_sin_mejora >= PACIENCIA:
            print(f"\n⏹  Early stopping activado en época {epoca} "
                  f"(sin mejora durante {PACIENCIA} épocas)")
            break

# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUACIÓN FINAL CON EL MEJOR MODELO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Cargando mejor modelo para evaluación final...")
modelo.load_state_dict(torch.load('mejor_modelo.pt'))

val_loss_final, val_acc_final = evaluar(semanas_val)
print(f"  Val loss final : {val_loss_final:.4f}")
print(f"  Val acc@{TOP_K}     : {val_acc_final:.2%}")
print("=" * 70)
print("¡Entrenamiento finalizado!")