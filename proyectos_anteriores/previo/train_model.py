"""
PyTorch Predictive Model for Weekly Task Scheduling — v4
========================================================
Predicts task_name, start_time and end_time for a full week.

Key insight: (prev_3_tasks, day_of_week, slot_in_day) predicts the
next task with 98.6% accuracy as a lookup table. But a lookup table
can't generalise to unseen combinations.

This model uses a SLIDING WINDOW of N previous tasks as explicit
input features (not through recurrence), giving ~5000 training samples
instead of just 45 week-sequences.

Architecture: Deep MLP with task-embedding concatenation
Each sample = one prediction step with its N-gram context.

Train/Val: weeks 0..44 / weeks 45..51 (no overlap)
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

CONTEXT_SIZE = 5   # Number of previous tasks to use as context

# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================

def load_and_preprocess(path: str = "aux_databse.json"):
    with open(path, "r") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"]   = pd.to_datetime(df["end_time"])

    df["day_of_week"]   = df["start_time"].dt.dayofweek
    df["start_hour"]    = df["start_time"].dt.hour
    df["start_minute"]  = df["start_time"].dt.minute
    df["end_hour"]      = df["end_time"].dt.hour
    df["end_minute"]    = df["end_time"].dt.minute
    df["duration_mins"] = (
        (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0
    )

    le = LabelEncoder()
    df["task_id"] = le.fit_transform(df["task_name"])
    num_classes = len(le.classes_)

    df = df.sort_values("start_time").reset_index(drop=True)
    new_week = df["day_of_week"] < df["day_of_week"].shift(1)
    new_week.iloc[0] = False
    df["week_id"] = new_week.cumsum()
    df["slot_in_day"]  = df.groupby(["week_id", "day_of_week"]).cumcount()
    df["slot_in_week"] = df.groupby("week_id").cumcount()

    return df, le, num_classes


# ============================================================
# 2. SLIDING WINDOW DATASET
# ============================================================

def build_ngram_samples(df, context_size, num_classes):
    """
    For each task in the dataset, build a sample:
      Input:  context of N previous tasks (within same week) + current features
      Target: current task_id + time components

    Samples are built PER TASK, not per week, giving many more training samples.
    """
    PAD_TASK = num_classes  # special padding ID

    samples = []
    for wid in range(df["week_id"].max() + 1):
        wdf = df[df["week_id"] == wid].reset_index(drop=True)
        if len(wdf) < 2:
            continue

        for i in range(1, len(wdf)):
            # N-gram context: previous tasks (padded if not enough)
            ctx_tasks = []
            ctx_days  = []
            ctx_times = []
            for j in range(context_size, 0, -1):
                idx = i - j
                if idx >= 0:
                    row = wdf.iloc[idx]
                    ctx_tasks.append(int(row["task_id"]))
                    ctx_days.append(int(row["day_of_week"]))
                    ctx_times.append([
                        row["start_hour"], row["start_minute"],
                        row["end_hour"], row["end_minute"],
                        row["duration_mins"],
                    ])
                else:
                    ctx_tasks.append(PAD_TASK)
                    ctx_days.append(0)
                    ctx_times.append([0.0, 0.0, 0.0, 0.0, 0.0])

            # Current row features
            cur = wdf.iloc[i]
            target_task = int(cur["task_id"])
            target_day  = int(cur["day_of_week"])
            target_time = [
                cur["start_hour"], cur["start_minute"],
                cur["end_hour"], cur["end_minute"],
            ]

            # Also include the immediately previous task's features as "last known"
            prev = wdf.iloc[i-1]

            samples.append({
                "ctx_tasks":    ctx_tasks,                          # (C,)
                "ctx_days":     ctx_days,                           # (C,)
                "ctx_times":    np.array(ctx_times, dtype=np.float32).flatten(),  # (C*5,)
                "cur_day":      target_day,
                "cur_slot_day": int(cur["slot_in_day"]),
                "cur_slot_week": int(cur["slot_in_week"]),
                "prev_end_h":   prev["end_hour"],
                "prev_end_m":   prev["end_minute"],
                "target_task":  target_task,
                "target_day":   target_day,
                "target_time":  np.array(target_time, dtype=np.float32),
                "week_id":      wid,
            })

    return samples


class NGramDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "ctx_tasks":     torch.tensor(s["ctx_tasks"], dtype=torch.long),
            "ctx_days":      torch.tensor(s["ctx_days"],  dtype=torch.long),
            "ctx_times":     torch.tensor(s["ctx_times"], dtype=torch.float32),
            "cur_day":       torch.tensor(s["cur_day"],   dtype=torch.long),
            "cur_slot_day":  torch.tensor(s["cur_slot_day"],  dtype=torch.float32),
            "cur_slot_week": torch.tensor(s["cur_slot_week"], dtype=torch.float32),
            "prev_end_h":    torch.tensor(s["prev_end_h"],    dtype=torch.float32),
            "prev_end_m":    torch.tensor(s["prev_end_m"],    dtype=torch.float32),
            "target_task":   torch.tensor(s["target_task"], dtype=torch.long),
            "target_day":    torch.tensor(s["target_day"],  dtype=torch.long),
            "target_time":   torch.tensor(s["target_time"], dtype=torch.float32),
        }


# ============================================================
# 3. MODEL — N-gram Context Network
# ============================================================

class NGramTaskPredictor(nn.Module):
    """
    Takes N previous tasks (embedded) + positional/temporal features
    and predicts the next task + time.

    Each previous task is embedded independently, then all embeddings
    are concatenated with positional features and passed through a
    deep residual MLP.
    """
    def __init__(self, num_classes, context_size=5, d_model=512, dropout=0.15):
        super().__init__()
        self.num_classes  = num_classes
        self.context_size = context_size

        # Embeddings per context position
        self.task_emb = nn.Embedding(num_classes + 1, 64)   # +1 for pad
        self.day_emb  = nn.Embedding(7, 16)

        # Per-context-position projection (so the model knows which position
        # each previous task was in)
        self.pos_emb = nn.Embedding(context_size, 16)

        # Context time features projection
        self.ctx_time_proj = nn.Sequential(
            nn.Linear(context_size * 5, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

        # Current position features projection
        self.cur_proj = nn.Sequential(
            nn.Linear(4, 32),  # slot_day, slot_week, prev_end_h, prev_end_m
            nn.GELU(),
            nn.Linear(32, 32),
        )

        # Total: C*(64+16+16) + 64 + 16(cur_day) + 32 = ?
        # C=5: 5*96 + 64 + 16 + 32 = 592
        feat_dim = context_size * (64 + 16 + 16) + 64 + 16 + 32

        # Deep residual MLP
        self.input_proj = nn.Linear(feat_dim, d_model)

        self.blocks = nn.ModuleList([
            ResBlock(d_model, dropout) for _ in range(6)
        ])

        # Heads
        self.task_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self.day_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 7),
        )

        self.time_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4),
        )

    def forward(self, ctx_tasks, ctx_days, ctx_times, cur_day,
                cur_slot_day, cur_slot_week, prev_end_h, prev_end_m):
        """
        ctx_tasks: (B, C) long
        ctx_days:  (B, C) long
        ctx_times: (B, C*5) float
        cur_day:   (B,) long
        cur_slot_day/week: (B,) float
        prev_end_h/m: (B,) float
        """
        B = ctx_tasks.size(0)
        C = self.context_size
        device = ctx_tasks.device

        # Embed each context position
        te = self.task_emb(ctx_tasks)     # (B, C, 64)
        de = self.day_emb(ctx_days)       # (B, C, 16)
        pos_idx = torch.arange(C, device=device).unsqueeze(0).expand(B, -1)
        pe = self.pos_emb(pos_idx)        # (B, C, 16)

        ctx_emb = torch.cat([te, de, pe], dim=-1)  # (B, C, 96)
        ctx_flat = ctx_emb.reshape(B, -1)           # (B, C*96)

        # Context time features
        ct = self.ctx_time_proj(ctx_times)           # (B, 64)

        # Current position
        cur_day_e = self.day_emb(cur_day)             # (B, 16)
        cur_feats = torch.stack([cur_slot_day, cur_slot_week,
                                  prev_end_h, prev_end_m], dim=-1)  # (B, 4)
        cur_p = self.cur_proj(cur_feats)              # (B, 32)

        # Concatenate all
        x = torch.cat([ctx_flat, ct, cur_day_e, cur_p], dim=-1)  # (B, feat_dim)
        x = self.input_proj(x)  # (B, d_model)

        for block in self.blocks:
            x = block(x)

        return self.task_head(x), self.day_head(x), self.time_head(x)


class ResBlock(nn.Module):
    def __init__(self, d, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


# ============================================================
# 4. TRAINING
# ============================================================

def train_model():
    TRAIN_WEEKS = 45
    EPOCHS      = 300
    PATIENCE    = 40
    LR          = 1e-3
    BATCH_SIZE  = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  PyTorch Task Predictor v4 — N-gram Context Network")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Context window: {CONTEXT_SIZE} previous tasks\n")

    # ---- Load ----
    df, le, num_classes = load_and_preprocess()
    total_weeks = df["week_id"].max() + 1

    print(f"  Total tasks:       {len(df)}")
    print(f"  Unique task names: {num_classes}")
    print(f"  Total weeks:       {total_weeks}\n")

    # ---- Normalisation stats ----
    train_df = df[df["week_id"] < TRAIN_WEEKS].copy()
    time_cols = ["start_hour", "start_minute", "end_hour", "end_minute"]
    time_mean = train_df[time_cols].mean().values.astype(np.float32)
    time_std  = train_df[time_cols].std().values.astype(np.float32)
    time_std  = np.clip(time_std, 1e-6, None)
    time_mean_t = torch.tensor(time_mean).to(DEVICE)
    time_std_t  = torch.tensor(time_std).to(DEVICE)

    cont_cols = ["slot_in_day", "slot_in_week", "end_hour", "end_minute"]
    cont_mean = train_df[cont_cols].mean().values.astype(np.float32)
    cont_std  = train_df[cont_cols].std().values.astype(np.float32)
    cont_std  = np.clip(cont_std, 1e-6, None)

    # Build normalised samples
    # Normalise time fields for context times and targets
    time_fields_ctx = ["start_hour", "start_minute", "end_hour", "end_minute", "duration_mins"]
    ctx_mean = train_df[time_fields_ctx].mean().values.astype(np.float32)
    ctx_std  = train_df[time_fields_ctx].std().values.astype(np.float32)
    ctx_std  = np.clip(ctx_std, 1e-6, None)

    # Apply normalisation to df before building samples
    for i, c in enumerate(time_fields_ctx):
        df[c] = (df[c] - ctx_mean[i]) / ctx_std[i]

    # Normalise slots
    slot_mean = train_df[["slot_in_day", "slot_in_week"]].mean().values.astype(np.float32)
    slot_std  = train_df[["slot_in_day", "slot_in_week"]].std().values.astype(np.float32)
    slot_std  = np.clip(slot_std, 1e-6, None)
    for i, c in enumerate(["slot_in_day", "slot_in_week"]):
        df[c] = (df[c] - slot_mean[i]) / slot_std[i]

    # ---- Build samples ----
    all_samples = build_ngram_samples(df, CONTEXT_SIZE, num_classes)
    train_samples = [s for s in all_samples if s["week_id"] < TRAIN_WEEKS]
    val_samples   = [s for s in all_samples if s["week_id"] >= TRAIN_WEEKS]

    print(f"  Training samples:   {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}\n")

    train_dl = DataLoader(NGramDataset(train_samples), batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=False)
    val_dl   = DataLoader(NGramDataset(val_samples),   batch_size=BATCH_SIZE,
                          shuffle=False)

    # ---- Model ----
    model = NGramTaskPredictor(
        num_classes=num_classes,
        context_size=CONTEXT_SIZE,
        d_model=512,
        dropout=0.15,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}\n")

    # ---- Loss ----
    ce_task = nn.CrossEntropyLoss()
    ce_day  = nn.CrossEntropyLoss()
    mse_fn  = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS,
        steps_per_epoch=len(train_dl), pct_start=0.1
    )

    best_task_acc = 0.0
    no_improve    = 0
    best_state    = None

    print(f"  {'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  "
          f"{'TaskAcc':>8}  {'DayAcc':>7}  {'StMAE':>7}  {'EnMAE':>7}")
    print("  " + "-" * 58)

    for epoch in range(1, EPOCHS + 1):
        # ---- TRAIN ----
        model.train()
        tr_sum, n_tr = 0.0, 0
        for batch in train_dl:
            ct = batch["ctx_tasks"].to(DEVICE)
            cd = batch["ctx_days"].to(DEVICE)
            ctime = batch["ctx_times"].to(DEVICE)
            cday = batch["cur_day"].to(DEVICE)
            csd = batch["cur_slot_day"].to(DEVICE)
            csw = batch["cur_slot_week"].to(DEVICE)
            peh = batch["prev_end_h"].to(DEVICE)
            pem = batch["prev_end_m"].to(DEVICE)
            tt = batch["target_task"].to(DEVICE)
            td = batch["target_day"].to(DEVICE)
            ttime = batch["target_time"].to(DEVICE)

            tl, dl_, tp = model(ct, cd, ctime, cday, csd, csw, peh, pem)

            loss = ce_task(tl, tt) + 0.3 * ce_day(dl_, td) + 0.5 * mse_fn(tp, ttime)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            tr_sum += loss.item()
            n_tr += 1

        avg_tr = tr_sum / max(n_tr, 1)

        # ---- VALIDATE ----
        model.eval()
        va_sum, n_va = 0.0, 0
        t_correct, d_correct, t_total = 0, 0, 0
        sum_sae, sum_eae = 0.0, 0.0

        with torch.no_grad():
            for batch in val_dl:
                ct = batch["ctx_tasks"].to(DEVICE)
                cd = batch["ctx_days"].to(DEVICE)
                ctime = batch["ctx_times"].to(DEVICE)
                cday = batch["cur_day"].to(DEVICE)
                csd = batch["cur_slot_day"].to(DEVICE)
                csw = batch["cur_slot_week"].to(DEVICE)
                peh = batch["prev_end_h"].to(DEVICE)
                pem = batch["prev_end_m"].to(DEVICE)
                tt = batch["target_task"].to(DEVICE)
                td = batch["target_day"].to(DEVICE)
                ttime = batch["target_time"].to(DEVICE)

                tl, dl_, tp = model(ct, cd, ctime, cday, csd, csw, peh, pem)

                loss = ce_task(tl, tt) + 0.3 * ce_day(dl_, td) + 0.5 * mse_fn(tp, ttime)
                va_sum += loss.item()
                n_va += 1

                preds_t = tl.argmax(-1)
                preds_d = dl_.argmax(-1)
                t_correct += (preds_t == tt).sum().item()
                d_correct += (preds_d == td).sum().item()
                t_total += tt.size(0)

                # Denorm time for MAE
                tp_de = tp * time_std_t + time_mean_t
                tt_de = ttime * time_std_t + time_mean_t
                sum_sae += torch.abs(
                    (tp_de[:,0]*60+tp_de[:,1]) - (tt_de[:,0]*60+tt_de[:,1])
                ).sum().item()
                sum_eae += torch.abs(
                    (tp_de[:,2]*60+tp_de[:,3]) - (tt_de[:,2]*60+tt_de[:,3])
                ).sum().item()

        avg_va = va_sum / max(n_va, 1)
        t_acc = t_correct / max(t_total, 1) * 100
        d_acc = d_correct / max(t_total, 1) * 100
        s_mae = sum_sae / max(t_total, 1)
        e_mae = sum_eae / max(t_total, 1)

        if epoch % 10 == 0 or epoch <= 5 or epoch == EPOCHS:
            print(
                f"  {epoch:4d}  {avg_tr:8.4f}  {avg_va:8.4f}  "
                f"{t_acc:7.2f}%  {d_acc:6.1f}%  {s_mae:6.1f}m  {e_mae:6.1f}m"
            )

        if t_acc > best_task_acc:
            best_task_acc = t_acc
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} — best: {best_task_acc:.2f}%")
                break

    model.load_state_dict(best_state)
    model.to(DEVICE)

    # ============================================================
    # 5. FINAL EVALUATION
    # ============================================================
    print("\n" + "=" * 70)
    print("  Final Results — Validation (weeks 45–51, teacher forcing)")
    print("=" * 70)

    model.eval()
    t_correct, d_correct, t_total = 0, 0, 0
    sum_sae, sum_eae = 0.0, 0.0
    topk_correct = {1: 0, 3: 0, 5: 0}
    all_results = []

    with torch.no_grad():
        for batch in val_dl:
            ct = batch["ctx_tasks"].to(DEVICE)
            cd = batch["ctx_days"].to(DEVICE)
            ctime = batch["ctx_times"].to(DEVICE)
            cday = batch["cur_day"].to(DEVICE)
            csd = batch["cur_slot_day"].to(DEVICE)
            csw = batch["cur_slot_week"].to(DEVICE)
            peh = batch["prev_end_h"].to(DEVICE)
            pem = batch["prev_end_m"].to(DEVICE)
            tt = batch["target_task"].to(DEVICE)
            td = batch["target_day"].to(DEVICE)
            ttime = batch["target_time"].to(DEVICE)

            tl, dl_, tp = model(ct, cd, ctime, cday, csd, csw, peh, pem)

            preds_t = tl.argmax(-1)
            t_correct += (preds_t == tt).sum().item()
            d_correct += (dl_.argmax(-1) == td).sum().item()
            t_total += tt.size(0)

            tp_de = tp * time_std_t + time_mean_t
            tt_de = ttime * time_std_t + time_mean_t
            sum_sae += torch.abs(
                (tp_de[:,0]*60+tp_de[:,1]) - (tt_de[:,0]*60+tt_de[:,1])
            ).sum().item()
            sum_eae += torch.abs(
                (tp_de[:,2]*60+tp_de[:,3]) - (tt_de[:,2]*60+tt_de[:,3])
            ).sum().item()

            for k in [1, 3, 5]:
                _, topk = tl.topk(k, dim=-1)
                topk_correct[k] += (topk == tt.unsqueeze(-1)).any(-1).sum().item()

            for i in range(tt.size(0)):
                all_results.append({
                    "pred_task": preds_t[i].item(),
                    "gt_task":   tt[i].item(),
                    "pred_time": tp_de[i].cpu().tolist(),
                    "gt_time":   tt_de[i].cpu().tolist(),
                    "gt_day":    td[i].item(),
                })

    t_acc = t_correct / max(t_total, 1) * 100
    d_acc = d_correct / max(t_total, 1) * 100
    s_mae = sum_sae / max(t_total, 1)
    e_mae = sum_eae / max(t_total, 1)

    print(f"\n  ┌──────────────────────────────────────────┐")
    print(f"  │  Task Name Accuracy:     {t_acc:6.2f}%         │")
    print(f"  │  Day of Week Accuracy:   {d_acc:6.2f}%         │")
    print(f"  │  Start Time MAE:         {s_mae:6.2f} min      │")
    print(f"  │  End Time MAE:           {e_mae:6.2f} min      │")
    print(f"  ├──────────────────────────────────────────┤")
    for k in [1, 3, 5]:
        ka = topk_correct[k] / max(t_total, 1) * 100
        print(f"  │  Top-{k} Accuracy:        {ka:6.2f}%         │")
    print(f"  └──────────────────────────────────────────┘")

    # ---- Per-week breakdown ----
    print(f"\n  Per-Week Breakdown:")
    print(f"  {'Week':>6}  {'Correct':>8}  {'Total':>6}  {'Accuracy':>9}")
    print("  " + "-" * 38)

    offset = 0
    for wid in range(TRAIN_WEEKS, total_weeks):
        week_samples = [s for s in val_samples if s["week_id"] == wid]
        n = len(week_samples)
        if n == 0:
            continue
        wr = all_results[offset:offset+n]
        c = sum(1 for r in wr if r["pred_task"] == r["gt_task"])
        print(f"  {wid:6d}  {c:8d}  {n:6d}  {c/max(n,1)*100:8.1f}%")
        offset += n

    # ---- Show first val week ----
    DOW = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

    def fmt(h, m):
        hi, mi = int(round(h)), int(round(m))
        return f"{max(0,min(23,hi)):02d}:{max(0,min(59,mi)):02d}"

    w45_samples = [s for s in val_samples if s["week_id"] == TRAIN_WEEKS]
    w45_results = all_results[:len(w45_samples)]

    print(f"\n" + "=" * 70)
    print(f"  Predicted vs Actual — Week 45 (primeras 30 tareas)")
    print("=" * 70)
    print(f"\n  {'#':>3}  {'✓':>1}  {'Predicted':<35}  {'Actual':<35}  "
          f"{'Day':>3}  {'PSt':>5}  {'ASt':>5}  {'PEn':>5}  {'AEn':>5}")
    print("  " + "-" * 110)

    correct_w, total_w = 0, 0
    for i, res in enumerate(w45_results):
        total_w += 1
        m = "✓" if res["pred_task"] == res["gt_task"] else "✗"
        if res["pred_task"] == res["gt_task"]:
            correct_w += 1

        if i < 30:  # Only show first 30
            pn = le.inverse_transform([res["pred_task"]])[0]
            gn = le.inverse_transform([res["gt_task"]])[0]
            dstr = DOW[res["gt_day"]]
            pt = res["pred_time"]
            gt = res["gt_time"]

            print(
                f"  {i+1:3d}  {m}  {pn:<35}  {gn:<35}  "
                f"{dstr:>3}  {fmt(pt[0],pt[1]):>5}  {fmt(gt[0],gt[1]):>5}  "
                f"{fmt(pt[2],pt[3]):>5}  {fmt(gt[2],gt[3]):>5}"
            )

    if total_w > 30:
        print(f"  ... ({total_w - 30} more tasks)")

    print(f"\n  Week 45 total: {correct_w}/{total_w} ({correct_w/max(total_w,1)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("  ¡Entrenamiento completado!")
    print("=" * 70)


if __name__ == "__main__":
    train_model()
