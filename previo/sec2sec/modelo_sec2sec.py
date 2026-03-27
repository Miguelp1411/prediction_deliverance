import torch
import torch.nn as nn
from dataset_3 import MAX_TASKS, N_MIN_BINS

N_NUMERIC_FEATS = 7  # sin_day, cos_day, sin_hour, cos_hour, sin_min, cos_min, duration


class WeekEncoder(nn.Module):
    def __init__(self, num_tasks, embed_dim=32, hidden_size=256, num_layers=2):
        super().__init__()
        self.task_embedding = nn.Embedding(num_tasks + 1, embed_dim, padding_idx=0)

        self.task_lstm = nn.LSTM(
            embed_dim + N_NUMERIC_FEATS, hidden_size, num_layers,  # ← antes era +4
            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0
        )
        self.week_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0
        )

    def forward(self, x, padding_mask):
        batch, num_weeks, max_tasks, _ = x.shape
        week_summaries = []

        for w in range(num_weeks):
            week_data = x[:, w, :, :]
            task_ids  = week_data[:, :, 0].long()
            num_feats = week_data[:, :, 1:]         # ahora son 7 columnas

            embedded = self.task_embedding(task_ids)
            inp      = torch.cat([embedded, num_feats], dim=-1)

            mask_w  = padding_mask[:, w, :]
            lengths = mask_w.sum(dim=1).clamp(min=1).cpu()

            packed = nn.utils.rnn.pack_padded_sequence(
                inp, lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.task_lstm(packed)
            week_summaries.append(hidden[-1])

        week_seq = torch.stack(week_summaries, dim=1)
        _, (context_h, context_c) = self.week_lstm(week_seq)
        return context_h, context_c


class WeekDecoder(nn.Module):
    def __init__(self, num_tasks, embed_dim=32, hidden_size=256, num_layers=2):
        super().__init__()
        self.task_embedding = nn.Embedding(num_tasks + 1, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim + N_NUMERIC_FEATS, hidden_size, num_layers,  # ← antes era +4
            batch_first=True, dropout=0.2 if num_layers > 1 else 0.0
        )

        self.head_task     = nn.Linear(hidden_size, num_tasks)
        self.head_day      = nn.Linear(hidden_size, 7)
        self.head_hour     = nn.Linear(hidden_size, 24)
        self.head_minute   = nn.Linear(hidden_size, N_MIN_BINS)
        self.head_duration = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden, cell):
        task_ids  = x[:, :, 0].long()
        num_feats = x[:, :, 1:]         # 7 features

        embedded = self.task_embedding(task_ids)
        inp      = torch.cat([embedded, num_feats], dim=-1)

        output, (h, c) = self.lstm(inp, (hidden, cell))

        return (
            self.head_task(output),
            self.head_day(output),
            self.head_hour(output),
            self.head_minute(output),
            self.head_duration(output).squeeze(-1),
            h, c
        )


class WeekPredictor(nn.Module):
    def __init__(self, num_tasks, embed_dim=32, hidden_size=256, num_layers=2):
        super().__init__()
        self.encoder = WeekEncoder(num_tasks, embed_dim, hidden_size, num_layers)
        self.decoder = WeekDecoder(num_tasks, embed_dim, hidden_size, num_layers)

    def forward(self, x, masks_X, target=None, teacher_forcing_ratio=1.0):
        context_h, context_c = self.encoder(x, masks_X)

        batch_size = x.shape[0]
        outputs = []

        decoder_input = torch.zeros(batch_size, 1, 8, device=x.device)  # ← 8 features
        h, c = context_h, context_c

        for t in range(MAX_TASKS):
            task_out, day_out, hour_out, min_out, dur_out, h, c = \
                self.decoder(decoder_input, h, c)
            outputs.append((task_out, day_out, hour_out, min_out, dur_out))

            use_teacher = (target is not None) and (torch.rand(1).item() < teacher_forcing_ratio)
            if use_teacher:
                decoder_input = target[:, t:t+1, :]
            else:
                import math
                pred_task_id  = (task_out[:, 0, :].argmax(-1).float() + 1)  # (batch,)
                pred_day_idx  = day_out[:, 0, :].argmax(-1).float()          # (batch,)
                pred_hour_idx = hour_out[:, 0, :].argmax(-1).float()         # (batch,)
                pred_min_idx  = min_out[:, 0, :].argmax(-1).float()          # (batch,)
                pred_dur      = dur_out[:, 0]                                 # (batch,)  ← sin unsqueeze

                sin_d = torch.sin(2 * math.pi * pred_day_idx / 7)
                cos_d = torch.cos(2 * math.pi * pred_day_idx / 7)
                sin_h = torch.sin(2 * math.pi * pred_hour_idx / 24)
                cos_h = torch.cos(2 * math.pi * pred_hour_idx / 24)
                sin_m = torch.sin(2 * math.pi * pred_min_idx / N_MIN_BINS)
                cos_m = torch.cos(2 * math.pi * pred_min_idx / N_MIN_BINS)

                # torch.stack sobre (batch,) → (batch, 8), luego unsqueeze(1) → (batch, 1, 8)
                decoder_input = torch.stack(
                    [pred_task_id, sin_d, cos_d, sin_h, cos_h, sin_m, cos_m, pred_dur],
                    dim=-1
                ).unsqueeze(1)

        task_all = torch.cat([o[0] for o in outputs], dim=1)
        day_all  = torch.cat([o[1] for o in outputs], dim=1)
        hour_all = torch.cat([o[2] for o in outputs], dim=1)
        min_all  = torch.cat([o[3] for o in outputs], dim=1)
        dur_all  = torch.cat([o[4] for o in outputs], dim=1)
        return task_all, day_all, hour_all, min_all, dur_all