import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, num_tareas_unicas, embedding_dim, hidden_size, num_layers=1, num_continuous_features=4):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. Capa Embedding: Convierte el task_id (entero) en un vector matemático
        self.embedding = nn.Embedding(num_tareas_unicas, embedding_dim)
        
        # La entrada a la GRU será la tarea + las 4 variables numéricas (día, hora, min, duración)
        gru_input_size = embedding_dim + num_continuous_features
        
        # 2. La GRU Simple
        self.gru = nn.GRU(
            input_size=gru_input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # 3. Capas de Salida (Dos cabezas)
        # Cabeza A: Predice qué tarea es la siguiente (Clasificación)
        self.fc_cat = nn.Linear(hidden_size, num_tareas_unicas)
        # Cabeza B: Predice el día, hora, min y duración de esa siguiente tarea (Regresión)
        self.fc_cont = nn.Linear(hidden_size, num_continuous_features)

    def forward(self, x_cat, x_cont, hidden=None):
        # x_cat shape: [batch_size, seq_len] (Secuencia de IDs de tareas)
        # x_cont shape: [batch_size, seq_len, 4] (Secuencia de datos de tiempo)
        
        # Pasamos los IDs por el embedding
        embedded = self.embedding(x_cat)
        
        # Concatenamos la tarea con su tiempo correspondiente en la dimensión de características (dim=2)
        # gru_input shape: [batch_size, seq_len, embedding_dim + 4]
        gru_input = torch.cat((embedded, x_cont), dim=2)
        
        # Pasamos la secuencia por la GRU
        # 'out' contiene las predicciones para cada paso temporal de la secuencia
        out, hidden = self.gru(gru_input, hidden)
        
        # Calculamos la predicción final pasando la salida de la GRU por nuestras capas lineales
        pred_cat = self.fc_cat(out)   # shape: [batch_size, seq_len, num_tareas_unicas]
        pred_cont = self.fc_cont(out) # shape: [batch_size, seq_len, 4]
        
        return pred_cat, pred_cont, hidden