import torch
import torch.nn as nn


class GRU(nn.Module):
    """
    GRU de doble cabeza (clasificación + regresión).

    Cambios respecto a la versión anterior:
      - hidden_size  64  → 32   (menos capacidad para evitar overfitting)
      - embedding_dim 32 → 16   (proporcional a hidden_size)
      - num_layers    2  → 1    (con tan pocos datos, 2 capas sobreajusta)
      - dropout       0.3 → 0.4 (más regularización para compensar)
    """

    def __init__(
        self,
        num_tareas_unicas: int,
        embedding_dim: int            = 16,   # reducido de 32
        hidden_size: int              = 32,   # reducido de 64
        num_layers: int               = 1,    # reducido de 2
        num_continuous_features: int  = 4,
        dropout: float                = 0.4,  # aumentado de 0.3
    ):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Embedding ────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(num_tareas_unicas, embedding_dim)

        gru_input_size = embedding_dim + num_continuous_features

        # ── GRU ──────────────────────────────────────────────────────────────
        # dropout interno solo actúa con num_layers > 1
        self.gru = nn.GRU(
            input_size  = gru_input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # ── Dropout antes de las cabezas ─────────────────────────────────────
        self.dropout = nn.Dropout(dropout)

        # ── Cabeza A: ¿qué tarea viene después? (clasificación) ──────────────
        self.fc_cat  = nn.Linear(hidden_size, num_tareas_unicas)

        # ── Cabeza B: ¿cuándo y cuánto dura? (regresión) ─────────────────────
        self.fc_cont = nn.Linear(hidden_size, num_continuous_features)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x_cat, x_cont, hidden=None):
        """
        x_cat  : [batch, seq_len]     — secuencia de task_ids
        x_cont : [batch, seq_len, 4]  — features continuas normalizadas
        hidden : estado oculto inicial (None = ceros)
        """
        embedded  = self.embedding(x_cat)               # [B, S, E]
        gru_input = torch.cat((embedded, x_cont), dim=2) # [B, S, E+4]

        out, hidden = self.gru(gru_input, hidden)         # [B, S, H]
        out         = self.dropout(out)

        pred_cat  = self.fc_cat(out)   # [B, S, num_tareas]
        pred_cont = self.fc_cont(out)  # [B, S, 4]

        return pred_cat, pred_cont, hidden


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICA: Top-K Accuracy
#
# CrossEntropyLoss dice cuánto error hay, pero no cuántas veces acertamos.
# Esta función mide si la tarea correcta está entre las K más probables.
# Con 50 tareas únicas, top-3 es una métrica mucho más útil que top-1.
# ─────────────────────────────────────────────────────────────────────────────
def top_k_accuracy(pred_logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    """
    Args:
        pred_logits : [N, num_tareas] — salida de fc_cat aplanada
        targets     : [N]             — etiquetas reales aplanadas
        k           : ventana (por defecto 3)

    Returns:
        float entre 0.0 y 1.0
    """
    k_real  = min(k, pred_logits.size(1))
    top_k   = torch.topk(pred_logits, k_real, dim=1).indices   # [N, k]
    correct = top_k.eq(targets.unsqueeze(1).expand_as(top_k))  # [N, k] bool
    return correct.any(dim=1).float().mean().item()