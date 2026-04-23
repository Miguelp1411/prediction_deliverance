import pandas as pd
import json
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────
with open('aux_databse.json', 'r') as archivo:
    datos_json = json.load(archivo)

df = pd.DataFrame(datos_json)
df_training = df[['task_name', 'start_time', 'end_time']].copy()

# ─────────────────────────────────────────────
# 2. INGENIERÍA DE CARACTERÍSTICAS
# ─────────────────────────────────────────────
df_training['start_time'] = pd.to_datetime(df_training['start_time'])
df_training['end_time']   = pd.to_datetime(df_training['end_time'])

df_training['duration_mins'] = (
    df_training['end_time'] - df_training['start_time']
).dt.total_seconds() / 60.0

df_training['day_of_week'] = df_training['start_time'].dt.dayofweek
df_training['hour']        = df_training['start_time'].dt.hour
df_training['minute']      = df_training['start_time'].dt.minute

# ─────────────────────────────────────────────
# 3. CODIFICACIÓN DE ETIQUETAS
# ─────────────────────────────────────────────
label_encoder = LabelEncoder()
df_training['task_id'] = label_encoder.fit_transform(df_training['task_name'])

# ─────────────────────────────────────────────
# 4. NORMALIZACIÓN DE FEATURES CONTINUAS
#    Sin esto, duration_mins (0-480) domina el MSE
#    y el gradiente ignora hour/minute/day_of_week
# ─────────────────────────────────────────────
COLS_CONTINUAS = ['day_of_week', 'hour', 'minute', 'duration_mins']

scaler = StandardScaler()
df_training[COLS_CONTINUAS] = scaler.fit_transform(df_training[COLS_CONTINUAS])

# ─────────────────────────────────────────────
# 5. DETECCIÓN DE SEMANAS
# ─────────────────────────────────────────────
df_training = df_training.sort_values('start_time').reset_index(drop=True)

# Usamos el día original (sin normalizar) para detectar el cambio de semana
dow_original            = df_training['start_time'].dt.dayofweek
es_nueva_semana         = dow_original < dow_original.shift(1)
es_nueva_semana.iloc[0] = False
df_training['week_id']  = es_nueva_semana.cumsum()

# ─────────────────────────────────────────────
# 6. SEPARACIÓN EN SEMANAS (mínimo 2 eventos)
# ─────────────────────────────────────────────
semanas_separadas = []
for i in range(df_training['week_id'].max() + 1):
    datos_semana = df_training[df_training['week_id'] == i]
    if len(datos_semana) >= 2:
        semanas_separadas.append(datos_semana)

# ─────────────────────────────────────────────
# 7. SPLIT TRAIN / VALIDATION
#    Las últimas semanas son validación (datos más recientes)
# ─────────────────────────────────────────────
N_VAL = max(2, int(len(semanas_separadas) * 0.15))  # 15% o mínimo 2

semanas_train = semanas_separadas[:-N_VAL]
semanas_val   = semanas_separadas[-N_VAL:]

print(f"Total semanas : {len(semanas_separadas)}")
print(f"  → Train     : {len(semanas_train)} semanas")
print(f"  → Val       : {len(semanas_val)} semanas")
print(f"Tareas únicas : {len(label_encoder.classes_)}")

# ─────────────────────────────────────────────
# 8. FUNCIÓN: secuencia completa (para validación)
# ─────────────────────────────────────────────
def semana_a_tensores(semana_df):
    """
    Convierte toda la semana en un único ejemplo de entrenamiento.
    Se usa principalmente en validación para evaluar sobre secuencias reales.
    """
    x_cat_data  = torch.tensor(semana_df['task_id'].values, dtype=torch.long)
    x_cont_data = torch.tensor(
        semana_df[COLS_CONTINUAS].values, dtype=torch.float32
    )
    X_cat  = x_cat_data[:-1].unsqueeze(0)   # [1, seq_len]
    X_cont = x_cont_data[:-1].unsqueeze(0)  # [1, seq_len, 4]
    Y_cat  = x_cat_data[1:].unsqueeze(0)    # [1, seq_len]
    Y_cont = x_cont_data[1:].unsqueeze(0)   # [1, seq_len, 4]
    return X_cat, X_cont, Y_cat, Y_cont


# ─────────────────────────────────────────────
# 9. FUNCIÓN: ventanas deslizantes (para entrenamiento)
#
#    DATA AUGMENTATION — combate el overfitting por falta de datos.
#
#    En lugar de usar la semana entera como 1 sola muestra,
#    genera múltiples subsecuencias de longitud fija 'seq_len'.
#
#    Ejemplo: semana con 30 eventos y seq_len=10
#      → 20 ventanas distintas en lugar de 1 secuencia de 29 pasos
#      → el dataset de train crece ~10-20x sin necesitar datos nuevos
# ─────────────────────────────────────────────
def semana_a_ventanas(semana_df, seq_len=10):
    """
    Args:
        semana_df : DataFrame de una semana ordenada cronológicamente
        seq_len   : número de eventos de entrada por ventana

    Returns:
        Lista de tuplas (X_cat, X_cont, Y_cat, Y_cont), una por ventana
    """
    x_cat_data  = torch.tensor(semana_df['task_id'].values, dtype=torch.long)
    x_cont_data = torch.tensor(
        semana_df[COLS_CONTINUAS].values, dtype=torch.float32
    )

    n = len(x_cat_data)

    # Semana más corta que seq_len → usamos la secuencia completa
    if n <= seq_len:
        return [(
            x_cat_data[:-1].unsqueeze(0),
            x_cont_data[:-1].unsqueeze(0),
            x_cat_data[1:].unsqueeze(0),
            x_cont_data[1:].unsqueeze(0),
        )]

    ventanas = []
    for i in range(n - seq_len):
        X_cat  = x_cat_data[i     : i + seq_len].unsqueeze(0)
        X_cont = x_cont_data[i    : i + seq_len].unsqueeze(0)
        Y_cat  = x_cat_data[i + 1 : i + seq_len + 1].unsqueeze(0)
        Y_cont = x_cont_data[i + 1 : i + seq_len + 1].unsqueeze(0)
        ventanas.append((X_cat, X_cont, Y_cat, Y_cont))

    return ventanas