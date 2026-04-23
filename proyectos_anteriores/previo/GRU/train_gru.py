import torch.optim as optim
import torch
import torch.nn as nn
from proyectos_anteriores.previo.dataset_2 import *
from proyectos_anteriores.previo.GRU.modelo_gru_2 import GRU

# 1. Instanciar el modelo
# Obtenemos el número total de tareas únicas desde tu LabelEncoder
num_tareas = len(label_encoder.classes_) 

modelo = GRU(
    num_tareas_unicas=num_tareas, 
    embedding_dim=16, 
    hidden_size=32
)

# 2. Definir funciones de error (Loss) y Optimizador
# CrossEntropy para adivinar la categoría (tarea)
criterio_clasificacion = nn.CrossEntropyLoss() 
# Mean Squared Error para adivinar los números continuos (tiempo)
criterio_regresion = nn.MSELoss()              

optimizador = optim.Adam(modelo.parameters(), lr=0.005)

# 3. Bucle de Entrenamiento
epocas = 150

print("Iniciando entrenamiento...\n")

for epoca in range(epocas):
    loss_total = 0.0
    
    # Iteramos directamente sobre tu lista de Pandas DataFrames
    for semana_df in semanas_separadas:
        
        # Si una semana tiene 1 sola tarea, no podemos hacer secuencia, la saltamos
        if len(semana_df) < 2:
            continue
            
        # A. Extraer datos del DataFrame de Pandas a Tensores de PyTorch
        x_cat_data = torch.tensor(semana_df['task_id'].values, dtype=torch.long)
        x_cont_data = torch.tensor(semana_df[['day_of_week', 'hour', 'minute', 'duration_mins']].values, dtype=torch.float32)
        
        # B. Hacer el desplazamiento (Shift) sobre la marcha y añadir dimensión de Batch (unsqueeze(0))
        # Entrada X (Lo que la red ve): Desde la primera tarea hasta la penúltima
        X_cat = x_cat_data[:-1].unsqueeze(0)   # Forma: [1, seq_len]
        X_cont = x_cont_data[:-1].unsqueeze(0) # Forma: [1, seq_len, 4]
        
        # Objetivo Y (Lo que debe predecir): Desde la segunda tarea hasta la última
        Y_cat = x_cat_data[1:].unsqueeze(0)    # Forma: [1, seq_len]
        Y_cont = x_cont_data[1:].unsqueeze(0)  # Forma: [1, seq_len, 4]
        
        # C. Entrenamiento (Paso hacia adelante)
        optimizador.zero_grad()
        
        pred_cat, pred_cont, _ = modelo(X_cat, X_cont)
        
        # D. Calcular el Error (Loss)
        # Para la clasificación, PyTorch requiere aplanar las dimensiones a 2D y 1D
        pred_cat_aplanado = pred_cat.view(-1, num_tareas)
        Y_cat_aplanado = Y_cat.view(-1)
        
        loss_cat = criterio_clasificacion(pred_cat_aplanado, Y_cat_aplanado)
        loss_cont = criterio_regresion(pred_cont, Y_cont)
        
        # Sumamos ambos errores (la red debe aprender a optimizar ambas cosas a la vez)
        loss = loss_cat + loss_cont
        
        # E. Paso hacia atrás y actualización de pesos
        loss.backward()
        optimizador.step()
        
        loss_total += loss.item()
        
    promedio_loss = loss_total / len(semanas_separadas)
    print(f"Época {epoca+1}/{epocas} completada | Pérdida (Loss) Promedio: {promedio_loss:.4f}")

print("\n¡Entrenamiento finalizado!")