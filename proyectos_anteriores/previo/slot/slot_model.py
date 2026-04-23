"""
slot_model.py
─────────────
RoutinePredictor: Transformer que predice slots en paralelo.

Arquitectura en dos etapas:

  1. TemporalEncoder:
       Para cada slot, mira su historia en las 4 semanas anteriores.
       (batch*K, WINDOW, d) → Transformer → representación temporal del slot

  2. ContextualEncoder:
       Los slots se miran entre sí. Captura co-ocurrencias:
       "si plancho los lunes, suelo fregar los martes"
       (batch, K, d) → Transformer → representación contextual

  3. Cabezas de predicción (paralelas, una por slot):
       - ¿Ocurre este slot la próxima semana?  (BCE)
       - ¿A qué hora?                          (CrossEntropy sobre 24 clases)
       - ¿En qué bin de minuto?                (CrossEntropy sobre 12 bins)
       - ¿Cuánto dura?                         (MSE)
"""

import torch
import torch.nn as nn
import math

N_MIN_BINS = 12


class TemporalEncoder(nn.Module):
    """
    Cada slot tiene una secuencia temporal de WINDOW observaciones.
    Este encoder aprende 'el patrón de lunes pasados para esta tarea'.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 dropout: float, window: int, input_dim: int = 6):
        super().__init__()
        self.window = window

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding aprendido para las semanas (0=antigua → window-1=reciente)
        self.week_pos = nn.Embedding(window, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model       = d_model,
            nhead         = n_heads,
            dim_feedforward = d_model * 4,
            dropout       = dropout,
            batch_first   = True,
            activation    = 'gelu',
            norm_first    = True,   # Pre-LN: más estable con datasets pequeños
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, WINDOW, K, input_dim)
        →  (batch, K, d_model)
        """
        batch, window, K, _ = x.shape

        # Proyectar features
        h = self.input_proj(x)   # (batch, WINDOW, K, d_model)

        # Añadir positional encoding de semana
        week_ids  = torch.arange(window, device=x.device)
        week_emb  = self.week_pos(week_ids)                           # (WINDOW, d_model)
        h = h + week_emb.unsqueeze(0).unsqueeze(2)                    # broadcast

        # Transformer temporal: (batch*K, WINDOW, d_model)
        h = h.permute(0, 2, 1, 3).contiguous().view(batch * K, window, -1)
        h = self.transformer(h)                    # (batch*K, WINDOW, d_model)

        # Tomamos la última posición temporal (la más reciente es la más predictiva)
        # + average pooling para robustez
        h = h[:, -1, :] * 0.6 + h.mean(dim=1) * 0.4

        return h.view(batch, K, -1)               # (batch, K, d_model)


class ContextualEncoder(nn.Module):
    """
    Encoder que permite a cada slot atender a todos los demás slots.
    Captura co-ocurrencias de tareas dentro de la semana.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            batch_first     = True,
            activation      = 'gelu',
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, K, d_model) → (batch, K, d_model)"""
        return self.transformer(x)


def _make_head(d_model: int, out_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(d_model, d_model // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model // 2, out_dim),
    )


class RoutinePredictor(nn.Module):
    """
    Modelo principal de predicción de rutinas semanales por slots.

    Parámetros recomendados para ~300 ventanas de entrenamiento:
      d_model=64, n_heads=4, n_layers=2, dropout=0.3
    """

    def __init__(
        self,
        num_slots : int,
        d_model   : int = 64,
        n_heads   : int = 4,
        n_layers  : int = 2,
        dropout   : float = 0.3,
        window    : int = 4,
        input_dim : int = 6,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.d_model   = d_model

        # ── Encoders ─────────────────────────────────────────────────────────
        self.temporal_enc   = TemporalEncoder(d_model, n_heads, n_layers,
                                               dropout, window, input_dim)
        self.contextual_enc = ContextualEncoder(d_model, n_heads, dropout)

        # Identidad del slot: le dice al modelo CUÁL slot es
        self.slot_embedding = nn.Embedding(num_slots, d_model)

        # Norma final antes de las cabezas
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # ── Cabezas de predicción ─────────────────────────────────────────────
        self.head_occurs  = _make_head(d_model, 1,          dropout)
        self.head_hour    = _make_head(d_model, 24,         dropout)
        self.head_minute  = _make_head(d_model, N_MIN_BINS, dropout)
        self.head_duration= _make_head(d_model, 1,          dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, WINDOW, K, 6)

        Returns:
          occurs   (batch, K)       — logits de ocurrencia
          hour     (batch, K, 24)   — logits de hora
          minute   (batch, K, 12)   — logits de bin de minuto
          duration (batch, K)       — duración normalizada
        """
        batch, _, K, _ = x.shape

        # 1. Codificación temporal por slot
        h = self.temporal_enc(x)                             # (batch, K, d_model)

        # 2. Añadir identidad del slot
        slot_ids = torch.arange(K, device=x.device)
        h = h + self.slot_embedding(slot_ids).unsqueeze(0)   # (batch, K, d_model)

        # 3. Codificación contextual (relaciones entre slots)
        h = self.contextual_enc(h)                           # (batch, K, d_model)

        h = self.drop(self.norm(h))

        # 4. Predicciones paralelas
        occurs   = self.head_occurs(h).squeeze(-1)           # (batch, K)
        hour     = self.head_hour(h)                         # (batch, K, 24)
        minute   = self.head_minute(h)                       # (batch, K, 12)
        duration = self.head_duration(h).squeeze(-1)         # (batch, K)

        return occurs, hour, minute, duration


def model_summary(model: nn.Module, num_slots: int):
    """Imprime un resumen conciso del modelo."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nRoutinePredictor")
    print(f"  Slots       : {num_slots}")
    print(f"  d_model     : {model.d_model}")
    print(f"  Parámetros  : {total:,}")
    print(f"  Paradigma   : Slot-based (no secuencial)")
    print()
