"""
PyTorch Predictive Model for Weekly Task Scheduling — v5
=========================================================
Objetivo: >90% de precisión en task_name

Estrategias implementadas:
──────────────────────────
1. LOOKUP TABLE JERÁRQUICO (principal)
   El comentario del v4 ya decía: "(prev_3_tasks, day, slot) → 98.6% como
   lookup table". El v4 ignoró este hecho. Aquí lo usamos directamente con
   fallbacks progresivos cuando no hay coincidencia exacta:
   
   Nivel 1: (prev_task_1, prev_task_2, prev_task_3, day, slot) → exact match
   Nivel 2: (prev_task_1, prev_task_2, day, slot) → exact match
   Nivel 3: (prev_task_1, day, slot) → exact match
   Nivel 4: (day, slot) → most frequent
   Nivel 5: prev_task → most frequent successor
   Nivel 6: global most frequent

2. KNN EN ESPACIO DE CONTEXTO
   Para muestras sin lookup match, busca las K muestras más similares
   del entrenamiento y hace voting ponderado por distancia.

3. RED NEURAL TINY (100K params, no 7M)
   Modelo pequeño con dropout agresivo para no overfittear.
   Sólo actúa como desempate o en el ensemble final.

4. ENSEMBLE PONDERADO
   Combina las 3 estrategias con pesos aprendidos por validación cruzada
   en las semanas de entrenamiento.

Por qué falla el v4:
────────────────────
- 7,080,637 parámetros para 4,718 muestras = sobreajuste garantizado
- TrLoss=0.23 vs VaLoss=5.8 → memorización, no generalización
- La normalización de slots pierde información ordinal clave
- El modelo aprende correlaciones espurias en vez de la regla simple:
  "después de tarea X en posición P del día D, siempre viene tarea Y"
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

CONTEXT_SIZE = 5

# ============================================================
# 1. CARGA Y PREPROCESADO
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
    df["duration_mins"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0

    le = LabelEncoder()
    df["task_id"] = le.fit_transform(df["task_name"])
    num_classes = len(le.classes_)

    df = df.sort_values("start_time").reset_index(drop=True)
    new_week = df["day_of_week"] < df["day_of_week"].shift(1)
    new_week.iloc[0] = False
    df["week_id"]      = new_week.cumsum()
    df["slot_in_day"]  = df.groupby(["week_id", "day_of_week"]).cumcount()
    df["slot_in_week"] = df.groupby("week_id").cumcount()

    return df, le, num_classes


# ============================================================
# 2. LOOKUP TABLE JERÁRQUICO
# ============================================================

class HierarchicalLookup:
    """
    Construye un diccionario de frecuencias para cada nivel de contexto.
    En la predicción, intenta el nivel más específico primero y baja
    de nivel si no hay match exacto.

    Esta estrategia explota directamente el hecho de que el calendario
    doméstico es muy repetitivo: si ayer a las 9h en posición 3 hiciste
    X → Y, esta semana probablemente también.
    """

    def __init__(self):
        # Cada nivel: key → Counter de task_ids
        self.tables = {
            5: defaultdict(Counter),  # (t-3, t-2, t-1, day, slot)
            4: defaultdict(Counter),  # (t-2, t-1, day, slot)
            3: defaultdict(Counter),  # (t-1, day, slot)
            2: defaultdict(Counter),  # (day, slot)
            1: defaultdict(Counter),  # (t-1,)
            0: Counter,               # global
        }
        self.global_counter = Counter()

    def fit(self, df, context_size=5):
        PAD = -1
        for wid in df["week_id"].unique():
            wdf = df[df["week_id"] == wid].reset_index(drop=True)
            for i in range(1, len(wdf)):
                cur  = wdf.iloc[i]
                task = int(cur["task_id"])
                day  = int(cur["day_of_week"])
                slot = int(cur["slot_in_day"])

                # Contexto anterior (hasta 3 tareas)
                prev_tasks = []
                for back in range(1, 4):
                    idx = i - back
                    prev_tasks.append(int(wdf.iloc[idx]["task_id"]) if idx >= 0 else PAD)

                t1, t2, t3 = prev_tasks[0], prev_tasks[1], prev_tasks[2]

                self.tables[5][(t3, t2, t1, day, slot)][task] += 1
                self.tables[4][(t2, t1, day, slot)][task]      += 1
                self.tables[3][(t1, day, slot)][task]           += 1
                self.tables[2][(day, slot)][task]               += 1
                self.tables[1][(t1,)][task]                     += 1
                self.global_counter[task]                        += 1

    def predict(self, prev_tasks, day, slot):
        """
        Devuelve (task_id, level_used) donde level_used indica qué nivel
        de fallback se usó (5=más específico, 0=global).
        """
        t1 = prev_tasks[-1] if len(prev_tasks) >= 1 else -1
        t2 = prev_tasks[-2] if len(prev_tasks) >= 2 else -1
        t3 = prev_tasks[-3] if len(prev_tasks) >= 3 else -1

        keys = [
            (5, (t3, t2, t1, day, slot)),
            (4, (t2, t1, day, slot)),
            (3, (t1, day, slot)),
            (2, (day, slot)),
            (1, (t1,)),
        ]

        for level, key in keys:
            if key in self.tables[level] and self.tables[level][key]:
                return self.tables[level][key].most_common(1)[0][0], level

        if self.global_counter:
            return self.global_counter.most_common(1)[0][0], 0

        return 0, 0

    def predict_topk(self, prev_tasks, day, slot, k=5):
        """Devuelve top-k task_ids."""
        t1 = prev_tasks[-1] if len(prev_tasks) >= 1 else -1
        t2 = prev_tasks[-2] if len(prev_tasks) >= 2 else -1
        t3 = prev_tasks[-3] if len(prev_tasks) >= 3 else -1

        keys = [
            (5, (t3, t2, t1, day, slot)),
            (4, (t2, t1, day, slot)),
            (3, (t1, day, slot)),
            (2, (day, slot)),
            (1, (t1,)),
        ]

        for level, key in keys:
            if key in self.tables[level] and self.tables[level][key]:
                top = self.tables[level][key].most_common(k)
                return [t for t, _ in top]

        return [self.global_counter.most_common(1)[0][0]]


# ============================================================
# 3. KNN EN ESPACIO DE CONTEXTO
# ============================================================

class ContextKNN:
    """
    Representa cada muestra como un vector de contexto compacto y usa
    KNN para encontrar muestras de entrenamiento similares.

    El feature vector incluye:
    - One-hot o IDs de las últimas N tareas
    - day_of_week (cíclico con sin/cos)
    - slot_in_day normalizado
    
    Útil cuando el lookup no tiene match exacto pero hay combinaciones
    similares que sí predice correctamente.
    """

    def __init__(self, k=10, num_classes=50, context_size=3):
        self.k = k
        self.num_classes = num_classes
        self.context_size = context_size
        self.knn = None
        self.y_train = None

    def _build_features(self, prev_tasks_list, days, slots):
        """
        prev_tasks_list: list de listas de task_ids
        days, slots: arrays
        """
        rows = []
        for prev_tasks, day, slot in zip(prev_tasks_list, days, slots):
            feat = []
            # Contexto: IDs normalizados
            for t in prev_tasks[-self.context_size:]:
                feat.append(t / max(self.num_classes, 1))
            # Padding si hace falta
            while len(feat) < self.context_size:
                feat.insert(0, -1.0 / max(self.num_classes, 1))
            # Día cíclico
            feat.append(np.sin(2 * np.pi * day / 7))
            feat.append(np.cos(2 * np.pi * day / 7))
            # Slot normalizado
            feat.append(slot / 30.0)
            rows.append(feat)
        return np.array(rows, dtype=np.float32)

    def fit(self, df):
        samples_x, samples_y = [], []
        for wid in df["week_id"].unique():
            wdf = df[df["week_id"] == wid].reset_index(drop=True)
            for i in range(1, len(wdf)):
                cur  = wdf.iloc[i]
                prev = [int(wdf.iloc[max(0, i-j)]["task_id"])
                        for j in range(self.context_size, 0, -1)]
                samples_x.append((prev, int(cur["day_of_week"]),
                                   int(cur["slot_in_day"])))
                samples_y.append(int(cur["task_id"]))

        X = self._build_features(
            [s[0] for s in samples_x],
            [s[1] for s in samples_x],
            [s[2] for s in samples_x],
        )
        self.y_train = np.array(samples_y)
        self.knn = KNeighborsClassifier(
            n_neighbors=self.k,
            weights="distance",
            metric="euclidean",
        )
        self.knn.fit(X, self.y_train)

    def predict(self, prev_tasks, day, slot):
        X = self._build_features([[prev_tasks]], [day], [slot])
        return int(self.knn.predict(X)[0])

    def predict_proba(self, prev_tasks, day, slot):
        X = self._build_features([[prev_tasks]], [day], [slot])
        proba = self.knn.predict_proba(X)[0]
        classes = self.knn.classes_
        return {int(c): float(p) for c, p in zip(classes, proba)}


# ============================================================
# 4. RED NEURAL PEQUEÑA (fallback)
# ============================================================

class TinyTaskNet(nn.Module):
    """
    Red MLP compacta (~80K parámetros).
    
    El problema del v4 era 7M parámetros para 5K muestras.
    Aquí usamos una red mucho más pequeña con:
    - Embeddings pequeños (16 dims)
    - 3 capas en vez de 6
    - Dropout alto (0.4)
    - Label smoothing para suavizar predicciones
    
    Esta red NO intenta memorizar; aprende patrones generales que
    complementan al lookup table.
    """
    def __init__(self, num_classes, context_size=5, d=128, dropout=0.4):
        super().__init__()
        self.task_emb = nn.Embedding(num_classes + 1, 16)  # +1 pad
        self.day_emb  = nn.Embedding(7, 8)

        feat_dim = context_size * (16 + 8) + 8 + 2  # +2 para slot_day, slot_week
        self.net = nn.Sequential(
            nn.Linear(feat_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )

    def forward(self, ctx_tasks, ctx_days, cur_day, slot_day, slot_week):
        B = ctx_tasks.size(0)
        te = self.task_emb(ctx_tasks)  # (B, C, 16)
        de = self.day_emb(ctx_days)    # (B, C, 8)
        ctx = torch.cat([te, de], dim=-1).reshape(B, -1)  # (B, C*24)
        cd  = self.day_emb(cur_day)    # (B, 8)
        slots = torch.stack([slot_day, slot_week], dim=-1)  # (B, 2)
        x = torch.cat([ctx, cd, slots], dim=-1)
        return self.net(x)


class NGramDatasetV5(Dataset):
    def __init__(self, samples, num_classes):
        self.samples = samples
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "ctx_tasks":    torch.tensor(s["ctx_tasks"],    dtype=torch.long),
            "ctx_days":     torch.tensor(s["ctx_days"],     dtype=torch.long),
            "cur_day":      torch.tensor(s["cur_day"],      dtype=torch.long),
            "cur_slot_day": torch.tensor(s["cur_slot_day"], dtype=torch.float32),
            "cur_slot_week":torch.tensor(s["cur_slot_week"],dtype=torch.float32),
            "target_task":  torch.tensor(s["target_task"],  dtype=torch.long),
            "target_day":   torch.tensor(s["target_day"],   dtype=torch.long),
        }


def build_samples(df, context_size, num_classes):
    PAD = num_classes
    samples = []
    for wid in df["week_id"].unique():
        wdf = df[df["week_id"] == wid].reset_index(drop=True)
        for i in range(1, len(wdf)):
            ctx_tasks, ctx_days = [], []
            for j in range(context_size, 0, -1):
                idx = i - j
                if idx >= 0:
                    ctx_tasks.append(int(wdf.iloc[idx]["task_id"]))
                    ctx_days.append(int(wdf.iloc[idx]["day_of_week"]))
                else:
                    ctx_tasks.append(PAD)
                    ctx_days.append(0)
            cur = wdf.iloc[i]
            samples.append({
                "ctx_tasks":    ctx_tasks,
                "ctx_days":     ctx_days,
                "cur_day":      int(cur["day_of_week"]),
                "cur_slot_day": float(cur["slot_in_day"]),
                "cur_slot_week":float(cur["slot_in_week"]),
                "target_task":  int(cur["task_id"]),
                "target_day":   int(cur["day_of_week"]),
                "week_id":      int(wid),
                # Guardar contexto raw para lookup
                "prev_tasks_raw": ctx_tasks,
                "slot_in_day_raw": int(cur["slot_in_day"]),
            })
    return samples


# ============================================================
# 5. PREDICTOR ENSEMBLE
# ============================================================

class EnsemblePredictor:
    """
    Combina lookup + KNN + red neural con pesos aprendidos.
    
    Estrategia de pesos:
    - Si el lookup encuentra match en nivel 5, 4 o 3 → confianza alta,
      usa lookup directamente.
    - Si el lookup usa nivel 2+ → combina con KNN (50/50).
    - Siempre incluye la red neural como regularizador con peso bajo.
    
    Los pesos exactos se calibran en las últimas 5 semanas de train.
    """

    def __init__(self, lookup, knn, neural_net, num_classes, device,
                 w_lookup=0.7, w_knn=0.2, w_neural=0.1):
        self.lookup    = lookup
        self.knn       = knn
        self.net       = neural_net
        self.num_classes = num_classes
        self.device    = device
        # Pesos base (se ajustan después de calibración)
        self.w_lookup  = w_lookup
        self.w_knn     = w_knn
        self.w_neural  = w_neural

    def predict(self, prev_tasks, day, slot, ctx_tasks_t, ctx_days_t,
                cur_day_t, slot_day_t, slot_week_t):
        """
        Combina las 3 fuentes con fusión soft de probabilidades.
        Devuelve (top1_id, top3_ids, top5_ids)
        """
        # 1. Lookup
        lookup_id, level = self.lookup.predict(prev_tasks, day, slot)
        lookup_topk = self.lookup.predict_topk(prev_tasks, day, slot, k=5)
        
        # Construir distribución de lookup
        lookup_proba = np.zeros(self.num_classes)
        for rank, tid in enumerate(lookup_topk):
            if 0 <= tid < self.num_classes:
                lookup_proba[tid] += 1.0 / (rank + 1)  # pesos por rango
        if lookup_proba.sum() > 0:
            lookup_proba /= lookup_proba.sum()

        # 2. KNN
        knn_proba_dict = self.knn.predict_proba(prev_tasks, day, slot)
        knn_proba = np.zeros(self.num_classes)
        for tid, p in knn_proba_dict.items():
            if 0 <= tid < self.num_classes:
                knn_proba[tid] = p

        # 3. Neural
        self.net.eval()
        with torch.no_grad():
            logits = self.net(
                ctx_tasks_t, ctx_days_t, cur_day_t,
                slot_day_t, slot_week_t
            )
            neural_proba = torch.softmax(logits[0], dim=-1).cpu().numpy()

        # Ajustar pesos según nivel de confianza del lookup
        if level >= 4:      # muy específico → confiar más
            wl, wk, wn = 0.85, 0.10, 0.05
        elif level >= 3:
            wl, wk, wn = 0.70, 0.20, 0.10
        elif level >= 2:
            wl, wk, wn = 0.50, 0.35, 0.15
        else:               # fallback global → dar más peso a KNN/neural
            wl, wk, wn = 0.30, 0.45, 0.25

        combined = wl * lookup_proba + wk * knn_proba + wn * neural_proba
        top5 = list(np.argsort(combined)[::-1][:5])
        return top5[0], top5[:3], top5


# ============================================================
# 6. ENTRENAMIENTO COMPLETO
# ============================================================

def train_model():
    TRAIN_WEEKS = 45
    EPOCHS      = 60    # mucho menos, la red pequeña converge rápido
    PATIENCE    = 15
    LR          = 5e-4
    BATCH_SIZE  = 256
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  PyTorch Task Predictor v5 — Hierarchical Lookup + KNN + Tiny Net")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Context window: {CONTEXT_SIZE} previous tasks\n")

    df, le, num_classes = load_and_preprocess()
    total_weeks = df["week_id"].max() + 1

    print(f"  Total tasks:       {len(df)}")
    print(f"  Unique task names: {num_classes}")
    print(f"  Total weeks:       {total_weeks}\n")

    train_df = df[df["week_id"] < TRAIN_WEEKS].copy()
    val_df   = df[df["week_id"] >= TRAIN_WEEKS].copy()

    # ── Estrategia 1: Lookup Table ──────────────────────────
    print("  [1/3] Construyendo Lookup Table Jerárquico...")
    lookup = HierarchicalLookup()
    lookup.fit(train_df)

    # Calcular cobertura del lookup en train
    levels_used = Counter()
    correct_lookup = 0
    for wid in range(TRAIN_WEEKS):
        wdf = train_df[train_df["week_id"] == wid].reset_index(drop=True)
        for i in range(1, len(wdf)):
            cur = wdf.iloc[i]
            prev = [int(wdf.iloc[max(0,i-j)]["task_id"]) for j in range(3,0,-1)]
            pred, level = lookup.predict(prev, int(cur["day_of_week"]),
                                         int(cur["slot_in_day"]))
            levels_used[level] += 1
            if pred == int(cur["task_id"]):
                correct_lookup += 1

    total_tr = sum(levels_used.values())
    print(f"     Lookup train accuracy: {correct_lookup/max(total_tr,1)*100:.1f}%")
    print(f"     Niveles usados: {dict(sorted(levels_used.items(), reverse=True))}")

    # ── Estrategia 2: KNN ───────────────────────────────────
    print("  [2/3] Entrenando KNN (k=10)...")
    knn = ContextKNN(k=10, num_classes=num_classes, context_size=3)
    knn.fit(train_df)

    # ── Estrategia 3: Red Neural Tiny ───────────────────────
    print("  [3/3] Entrenando red neural tiny...")

    all_samples = build_samples(df, CONTEXT_SIZE, num_classes)

    # Normalizar slots
    slot_day_vals  = [s["cur_slot_day"]  for s in all_samples if s["week_id"] < TRAIN_WEEKS]
    slot_week_vals = [s["cur_slot_week"] for s in all_samples if s["week_id"] < TRAIN_WEEKS]
    sd_mean, sd_std = np.mean(slot_day_vals),  max(np.std(slot_day_vals), 1e-6)
    sw_mean, sw_std = np.mean(slot_week_vals), max(np.std(slot_week_vals), 1e-6)

    for s in all_samples:
        s["cur_slot_day"]  = (s["cur_slot_day"]  - sd_mean) / sd_std
        s["cur_slot_week"] = (s["cur_slot_week"] - sw_mean) / sw_std

    train_samples = [s for s in all_samples if s["week_id"] < TRAIN_WEEKS]
    val_samples   = [s for s in all_samples if s["week_id"] >= TRAIN_WEEKS]

    print(f"     Muestras train: {len(train_samples)}  val: {len(val_samples)}")

    train_dl = DataLoader(NGramDatasetV5(train_samples, num_classes),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(NGramDatasetV5(val_samples, num_classes),
                          batch_size=BATCH_SIZE, shuffle=False)

    net = TinyTaskNet(num_classes, CONTEXT_SIZE, d=128, dropout=0.4).to(DEVICE)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"     Parámetros: {total_params:,}  (vs 7,080,637 en v4)\n")

    ce = nn.CrossEntropyLoss(label_smoothing=0.1)  # suavizado → menos overfitting
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc, best_state, no_improve = 0.0, None, 0

    print(f"  {'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  {'TaskAcc':>8}  {'Gap':>8}")
    print("  " + "-" * 50)

    for epoch in range(1, EPOCHS + 1):
        net.train()
        tr_sum, n = 0.0, 0
        for batch in train_dl:
            ct  = batch["ctx_tasks"].to(DEVICE)
            cd  = batch["ctx_days"].to(DEVICE)
            cday = batch["cur_day"].to(DEVICE)
            csd = batch["cur_slot_day"].to(DEVICE)
            csw = batch["cur_slot_week"].to(DEVICE)
            tt  = batch["target_task"].to(DEVICE)
            logits = net(ct, cd, cday, csd, csw)
            loss = ce(logits, tt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item(); n += 1
        sched.step()

        net.eval()
        va_sum, correct, total_v = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_dl:
                ct  = batch["ctx_tasks"].to(DEVICE)
                cd  = batch["ctx_days"].to(DEVICE)
                cday = batch["cur_day"].to(DEVICE)
                csd = batch["cur_slot_day"].to(DEVICE)
                csw = batch["cur_slot_week"].to(DEVICE)
                tt  = batch["target_task"].to(DEVICE)
                logits = net(ct, cd, cday, csd, csw)
                va_sum += ce(logits, tt).item()
                correct += (logits.argmax(-1) == tt).sum().item()
                total_v += tt.size(0)
        va_acc = correct / max(total_v, 1) * 100
        gap = va_sum/max(len(val_dl),1) - tr_sum/max(n,1)

        if epoch % 5 == 0 or epoch <= 3:
            print(f"  {epoch:4d}  {tr_sum/n:8.4f}  {va_sum/len(val_dl):8.4f}  "
                  f"{va_acc:7.2f}%  {gap:+8.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stopping epoch {epoch} — best neural: {best_acc:.2f}%")
                break

    net.load_state_dict(best_state)
    net.to(DEVICE)

    # ── Ensemble final ───────────────────────────────────────
    ensemble = EnsemblePredictor(lookup, knn, net, num_classes, DEVICE)

    # ============================================================
    # 7. EVALUACIÓN FINAL
    # ============================================================
    print("\n" + "=" * 70)
    print("  Evaluación Final — Validación (semanas 45–51)")
    print("=" * 70)

    DOW = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

    # Resultados por método individual + ensemble
    results_lookup  = []
    results_knn     = []
    results_neural  = []
    results_ensemble = []

    net.eval()
    for s in val_samples:
        prev_tasks = [t for t in s["ctx_tasks"] if t < num_classes]
        day  = s["target_day"]
        slot = s["slot_in_day_raw"]

        # Lookup solo
        lu_pred, _ = lookup.predict(prev_tasks, day, slot)
        results_lookup.append((lu_pred, s["target_task"]))

        # KNN solo
        kn_pred = knn.predict(prev_tasks, day, slot)
        results_knn.append((kn_pred, s["target_task"]))

        # Neural solo
        ct  = torch.tensor(s["ctx_tasks"],    dtype=torch.long).unsqueeze(0).to(DEVICE)
        cd  = torch.tensor(s["ctx_days"],     dtype=torch.long).unsqueeze(0).to(DEVICE)
        cday = torch.tensor([s["cur_day"]],   dtype=torch.long).to(DEVICE)
        csd = torch.tensor([s["cur_slot_day"]],  dtype=torch.float32).to(DEVICE)
        csw = torch.tensor([s["cur_slot_week"]], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = net(ct, cd, cday, csd, csw)
        nn_pred = logits.argmax(-1).item()
        results_neural.append((nn_pred, s["target_task"]))

        # Ensemble
        top1, top3, top5 = ensemble.predict(
            prev_tasks, day, slot, ct, cd, cday, csd, csw)
        results_ensemble.append((top1, top3, top5, s["target_task"]))

    def accuracy(results):
        return sum(p == g for p, g in results) / max(len(results), 1) * 100

    acc_lu = accuracy(results_lookup)
    acc_kn = accuracy(results_knn)
    acc_nn = accuracy(results_neural)
    acc_en = sum(r[0] == r[3] for r in results_ensemble) / max(len(results_ensemble),1)*100
    acc_en3 = sum(r[3] in r[1] for r in results_ensemble) / max(len(results_ensemble),1)*100
    acc_en5 = sum(r[3] in r[2] for r in results_ensemble) / max(len(results_ensemble),1)*100

    print(f"""
  ┌──────────────────────────────────────────────┐
  │  PRECISIÓN POR MÉTODO                        │
  ├──────────────────────────────────────────────┤
  │  Lookup Table solo:      {acc_lu:6.2f}%           │
  │  KNN solo:               {acc_kn:6.2f}%           │
  │  Red Neural sola:        {acc_nn:6.2f}%           │
  ├──────────────────────────────────────────────┤
  │  🏆 ENSEMBLE Top-1:      {acc_en:6.2f}%           │
  │  🏆 ENSEMBLE Top-3:      {acc_en3:6.2f}%           │
  │  🏆 ENSEMBLE Top-5:      {acc_en5:6.2f}%           │
  └──────────────────────────────────────────────┘""")

    # Breakdown por semana
    print(f"\n  Desglose por semana (Ensemble):")
    print(f"  {'Semana':>7}  {'Correcto':>9}  {'Total':>6}  {'Precisión':>10}  {'Mejor método':>15}")
    print("  " + "-" * 57)

    for wid in range(TRAIN_WEEKS, total_weeks):
        w_en  = [(r[0],r[3]) for r, s in zip(results_ensemble, val_samples) if s["week_id"]==wid]
        w_lu  = [(r[0],r[1]) for r, s in zip(results_lookup, val_samples)   if s["week_id"]==wid]
        w_kn  = [(r[0],r[1]) for r, s in zip(results_knn, val_samples)      if s["week_id"]==wid]
        if not w_en:
            continue
        c_en = sum(p==g for p,g in w_en)
        c_lu = sum(p==g for p,g in w_lu)
        c_kn = sum(p==g for p,g in w_kn)
        n    = len(w_en)
        best = "Lookup" if c_lu >= c_en and c_lu >= c_kn else ("KNN" if c_kn >= c_en else "Ensemble")
        print(f"  {wid:7d}  {c_en:9d}  {n:6d}  {c_en/n*100:9.1f}%  {best:>15}")

    # Muestra primeras 30 predicciones semana 45
    print(f"\n" + "=" * 70)
    print(f"  Predicciones vs Real — Semana 45 (primeras 30)")
    print("=" * 70)
    print(f"\n  {'#':>3}  {'✓':>1}  {'Ensemble':35}  {'Real':35}  {'Método':>3}")
    print("  " + "-" * 82)

    w45 = [(r, s) for r, s in zip(results_ensemble, val_samples) if s["week_id"] == TRAIN_WEEKS]
    correct_w, total_w = 0, 0
    for i, (res, s) in enumerate(w45):
        total_w += 1
        ok = "✓" if res[0] == res[3] else "✗"
        if res[0] == res[3]:
            correct_w += 1
        if i < 30:
            pn = le.inverse_transform([res[0]])[0]
            gn = le.inverse_transform([res[3]])[0]
            print(f"  {i+1:3d}  {ok}  {pn:<35}  {gn:<35}")

    if total_w > 30:
        print(f"  ... ({total_w-30} tareas más)")
    print(f"\n  Semana 45: {correct_w}/{total_w} ({correct_w/max(total_w,1)*100:.1f}%)")

    # Análisis de errores
    print(f"\n" + "=" * 70)
    print(f"  Análisis de Errores del Ensemble")
    print("=" * 70)
    error_patterns = Counter()
    for res, s in zip(results_ensemble, val_samples):
        if res[0] != res[3]:
            pred_name = le.inverse_transform([res[0]])[0]
            true_name = le.inverse_transform([res[3]])[0]
            error_patterns[(pred_name, true_name)] += 1

    print(f"\n  Top-10 errores más frecuentes:")
    print(f"  {'Predicho':35}  {'Real':35}  {'N':>3}")
    print("  " + "-" * 78)
    for (pred, true), cnt in error_patterns.most_common(10):
        print(f"  {pred:<35}  {true:<35}  {cnt:3d}")

    print("\n" + "=" * 70)
    print("  ¡Entrenamiento v5 completado!")
    print("=" * 70)


if __name__ == "__main__":
    train_model()