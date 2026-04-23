import pandas as pd
import json
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder

# Cargamos los datos
with open('aux_databse.json', 'r') as archivo:
    datos_json = json.load(archivo)

df = pd.DataFrame(datos_json)
df_training = df[['task_name', 'start_time', 'end_time']].copy()

# Procesamos las fechas y características
df_training['start_time'] = pd.to_datetime(df_training['start_time'])
df_training['end_time']   = pd.to_datetime(df_training['end_time'])
df_training['duration_mins'] = (
    df_training['end_time'] - df_training['start_time']
).dt.total_seconds() / 60.0
df_training['day_of_week'] = df_training['start_time'].dt.dayofweek
df_training['hour']        = df_training['start_time'].dt.hour
df_training['minute']      = df_training['start_time'].dt.minute

# Codificamos las tareas a números
label_encoder = LabelEncoder()
df_training['task_id'] = label_encoder.fit_transform(df_training['task_name'])

# Aseguramos orden cronológico
df_training = df_training.sort_values('start_time').reset_index(drop=True)

# ── Sliding window: ventanas de 7 días desplazadas 1 día cada vez ─────────────
# En lugar de 52 semanas fijas obtenemos ~330 ventanas solapadas,
# lo que multiplica los ejemplos de entrenamiento por ~7x.
# El modelo sigue prediciendo SIEMPRE una semana completa (7 días).

MIN_TASKS_POR_VENTANA = 10   # descartamos ventanas casi vacías

semanas_separadas = []

fecha_inicio = df_training['start_time'].min().normalize()  # medianoche del primer día
fecha_fin    = df_training['start_time'].max()

inicio = fecha_inicio
while True:
    fin = inicio + timedelta(days=7)
    if fin > fecha_fin:
        break

    ventana = df_training[
        (df_training['start_time'] >= inicio) &
        (df_training['start_time'] <  fin)
    ].copy()

    if len(ventana) >= MIN_TASKS_POR_VENTANA:
        semanas_separadas.append(ventana)

    inicio += timedelta(days=1)   # avanzamos 1 día, no 7

print(f"Total de ventanas extraídas: {len(semanas_separadas)}")
print(f"(equivale a ~{len(semanas_separadas) // 7}x más datos que con semanas fijas)\n")