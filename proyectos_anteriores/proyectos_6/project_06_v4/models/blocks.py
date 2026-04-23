import torch
import torch.nn as nn


class SequenceContextEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_proj = nn.Linear(hidden_size * 4, hidden_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        outputs, hidden = self.rnn(x)
        attn_scores = self.attn(outputs).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(outputs * attn_weights, dim=1)
        last_forward = hidden[-2]
        last_backward = hidden[-1]
        last_hidden = torch.cat([last_forward, last_backward], dim=-1)
        combined = torch.cat([pooled, last_hidden], dim=-1)
        return self.out_proj(combined) + self.residual_proj(combined)

