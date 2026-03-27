import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

# Cargamos los datos
with open('aux_databse.json', 'r') as archivo:
    datos_json = json.load(archivo)

df = pd.DataFrame(datos_json)
df_training = df[['task_name', 'start_time', 'end_time']].copy()

# Procesamos las fechas y características
df_training['start_time'] = pd.to_datetime(df_training['start_time'])
df_training['end_time'] = pd.to_datetime(df_training['end_time'])
df_training['duration_mins'] = (df_training['end_time'] - df_training['start_time']).dt.total_seconds() / 60.0
df_training['day_of_week'] = df_training['start_time'].dt.dayofweek
df_training['hour'] = df_training['start_time'].dt.hour
df_training['minute'] = df_training['start_time'].dt.minute

# Codificamos las tareas a números
label_encoder = LabelEncoder()
df_training['task_id'] = label_encoder.fit_transform(df_training['task_name'])

# --- APLICANDO LA LÓGICA DE CAMBIO DE SEMANA ---

# 1. Aseguramos que todo esté ordenado cronológicamente
df_training = df_training.sort_values('start_time').reset_index(drop=True)

# 2. Condición: ¿Es el día actual menor que el anterior?
es_nueva_semana = df_training['day_of_week'] < df_training['day_of_week'].shift(1)
es_nueva_semana.iloc[0] = False

# 3. Creamos el ID secuencial para cada semana
df_training['week_id'] = es_nueva_semana.cumsum()

# --- NUEVA AGRUPACIÓN: SIMPLEMENTE SEMANA A SEMANA ---

semanas_separadas = [] # Aquí guardaremos una lista donde cada elemento es el DataFrame de una semana

total_semanas = df_training['week_id'].max()

# Iteramos desde la semana 0 hasta la última
for i in range(total_semanas + 1): 
    # Extraemos solo los datos de la semana 'i'
    datos_semana = df_training[df_training['week_id'] == i]
    
    # Si la semana tiene datos, la añadimos a nuestra lista
    if not datos_semana.empty:
        semanas_separadas.append(datos_semana)

print(f"Total de semanas extraídas: {len(semanas_separadas)}\n")

# --- MOSTRAR LAS DOS PRIMERAS SEMANAS ---

# Columnas que queremos visualizar para que sea legible en la consola
columnas_a_mostrar = ['task_id', 'task_name', 'day_of_week', 'hour', 'minute', 'duration_mins']

"""if len(semanas_separadas) > 0:
    print("================ SEMANA 0 ================")
    # Mostramos todas las filas de la semana 0 (o puedes poner .head(10) si son muchas)
    print(semanas_separadas[0][columnas_a_mostrar])
    print("\n")

if len(semanas_separadas) > 1:
    print("================ SEMANA 1 ================")
    print(semanas_separadas[1][columnas_a_mostrar])
    print("\n")"""