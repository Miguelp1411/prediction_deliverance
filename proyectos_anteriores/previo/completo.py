"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CARGADOR DE DATOS JSON  →  Lista de DataFrames Semanales      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Uso:
    from data_loader import cargar_semanas_desde_json, TaskEncoder

    weekly_dfs, encoder = cargar_semanas_desde_json("tasks.json")
    # weekly_dfs : List[pd.DataFrame]  — una entrada por semana ISO
    # encoder    : TaskEncoder         — para decodificar task_id → task_name

Formato JSON de entrada (array o JSON-Lines):
    [
      {"uid": "...", "task_name": "Fregar suelo entrada",
       "type": "Delivery", "status": "Scheduled",
       "start_time": "2026-12-31T20:49:00.000Z",
       "end_time":   "2026-12-31T21:14:00.000Z"},
      ...
    ]
"""

import json
import pathlib
import warnings
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# 1. CODIFICADOR DE NOMBRES DE TAREA
# ══════════════════════════════════════════════════════════════════════════════

class TaskEncoder:
    """
    Mapea task_name (str) ↔ task_id (int 0..N-1).

    Se construye a partir de los datos de entrenamiento y se puede
    serializar a JSON para reutilizarlo en inferencia sin reentrenar.

    Atributos:
        name_to_id : dict[str, int]
        id_to_name : dict[int, str]
        n_classes  : int
    """

    UNKNOWN_ID = -1  # tarea no vista en train; el modelo la ignora en inferencia

    def __init__(self):
        self.name_to_id: dict = {}
        self.id_to_name: dict = {}
        self.n_classes:  int  = 0

    def fit(self, task_names: pd.Series) -> "TaskEncoder":
        """
        Construye el vocabulario ordenando alfabéticamente para
        asegurar determinismo independientemente del orden de los datos.
        """
        unique_names = sorted(task_names.dropna().unique().tolist())
        self.name_to_id = {name: idx for idx, name in enumerate(unique_names)}
        self.id_to_name = {idx: name for name, idx in self.name_to_id.items()}
        self.n_classes  = len(unique_names)
        return self

    def transform(self, task_names: pd.Series) -> pd.Series:
        """Convierte nombres a ids; tareas desconocidas → UNKNOWN_ID."""
        return task_names.map(
            lambda n: self.name_to_id.get(n, self.UNKNOWN_ID)
        )

    def inverse_transform(self, task_ids: pd.Series) -> pd.Series:
        """Convierte ids a nombres; ids desconocidos → '<UNK>'."""
        return task_ids.map(lambda i: self.id_to_name.get(i, "<UNK>"))

    def save(self, path: str) -> None:
        """Guarda el vocabulario en JSON para reutilizarlo."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name_to_id": self.name_to_id}, f, ensure_ascii=False, indent=2)
        print(f"[TaskEncoder] Vocabulario guardado en '{path}'")

    @classmethod
    def load(cls, path: str) -> "TaskEncoder":
        """Carga el vocabulario desde un JSON guardado previamente."""
        enc = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        enc.name_to_id = {k: int(v) for k, v in data["name_to_id"].items()}
        enc.id_to_name = {v: k for k, v in enc.name_to_id.items()}
        enc.n_classes  = len(enc.name_to_id)
        print(f"[TaskEncoder] Vocabulario cargado: {enc.n_classes} tareas")
        return enc


# ══════════════════════════════════════════════════════════════════════════════
# 2. PARSING Y LIMPIEZA DEL JSON RAW
# ══════════════════════════════════════════════════════════════════════════════

def _parse_json_file(path: Union[str, pathlib.Path]) -> pd.DataFrame:
    """
    Lee el fichero JSON y devuelve un DataFrame raw con todas las tareas.

    Soporta dos formatos:
      · Array JSON  : [ {...}, {...}, ... ]
      · JSON-Lines  : una tarea por línea  { ... }\\n{ ... }\\n...
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el fichero: {path}")

    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()

    # ── Detectar formato ──────────────────────────────────────────────────────
    if raw.startswith("["):
        records = json.loads(raw)                          # Array JSON
    else:
        records = [json.loads(line)                        # JSON-Lines
                   for line in raw.splitlines() if line.strip()]

    df = pd.DataFrame(records)
    print(f"[Parser] {len(df):,} tareas leídas de '{path.name}'")
    return df


def _clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza y feature engineering sobre el DataFrame raw.

    Pasos:
      1. Convertir start_time / end_time a datetime UTC
      2. Eliminar filas con fechas nulas o end < start
      3. Calcular duration_mins
      4. Extraer day_of_week, hour, minute
      5. Redondear minute al cuarto de hora más cercano
      6. Eliminar duplicados por uid
      7. Ordenar cronológicamente
      8. Añadir columna 'iso_week' = (year, week ISO) para agrupar semanas
    """
    df = df.copy()

    # ── 1. Parsear fechas ─────────────────────────────────────────────────────
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df["end_time"]   = pd.to_datetime(df["end_time"],   utc=True, errors="coerce")

    # ── 2. Filtrar filas inválidas ────────────────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=["start_time", "end_time", "task_name"])
    df = df[df["end_time"] > df["start_time"]]   # end debe ser posterior a start
    n_dropped = n_before - len(df)
    if n_dropped:
        warnings.warn(f"[Limpieza] {n_dropped} filas eliminadas (fechas inválidas)")

    # ── 3. Duración ────────────────────────────────────────────────────────────
    df["duration_mins"] = (
        (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0
    )
    # Filtrar duraciones absurdas (< 1 min o > 8 horas)
    df = df[(df["duration_mins"] >= 1) & (df["duration_mins"] <= 480)]

    # ── 4. Features temporales ────────────────────────────────────────────────
    # day_of_week: 0 = lunes … 6 = domingo  (estándar Python/Pandas)
    df["day_of_week"] = df["start_time"].dt.dayofweek
    df["hour"]        = df["start_time"].dt.hour
    df["minute_raw"]  = df["start_time"].dt.minute

    # ── 5. Redondear minuto al cuarto de hora más cercano (0,15,30,45) ────────
    df["minute"] = (df["minute_raw"] / 15).round().astype(int) * 15
    df["minute"] = df["minute"].clip(0, 45)   # 60 → 45 por seguridad

    # ── 6. Eliminar duplicados por uid (si existe la columna) ─────────────────
    if "uid" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=["uid"])
        if len(df) < n_before:
            warnings.warn(f"[Limpieza] {n_before - len(df)} duplicados eliminados")

    # ── 7. Orden cronológico ──────────────────────────────────────────────────
    df = df.sort_values("start_time").reset_index(drop=True)

    # ── 8. Semana ISO para agrupación ─────────────────────────────────────────
    # iso_week = (año, número_semana_ISO)  →  garantiza que la semana 52/53
    # de un año no se mezcle con la semana 1 del siguiente.
    df["iso_year"] = df["start_time"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["start_time"].dt.isocalendar().week.astype(int)
    df["week_key"] = list(zip(df["iso_year"], df["iso_week"]))

    print(f"[Limpieza] {len(df):,} tareas válidas  "
          f"| rango: {df['start_time'].min().date()} → {df['start_time'].max().date()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. AGRUPACIÓN EN SEMANAS
# ══════════════════════════════════════════════════════════════════════════════

def _group_into_weeks(
    df: pd.DataFrame,
    encoder: TaskEncoder,
    min_tasks_per_week: int = 10,
) -> List[pd.DataFrame]:
    """
    Agrupa el DataFrame limpio en una lista de DataFrames semanales.

    Cada elemento de la lista corresponde a una semana ISO completa y
    contiene sólo las columnas que necesita el modelo:
      [task_id, day_of_week, hour, minute, duration_mins]

    Las semanas con menos de min_tasks_per_week tareas se descartan
    (probablemente semanas incompletas al inicio/fin del dataset).
    """
    # Codificar task_name → task_id
    df["task_id"] = encoder.transform(df["task_name"])

    # Eliminar tareas con task_id desconocido (UNKNOWN_ID = -1)
    unknown_mask = df["task_id"] == TaskEncoder.UNKNOWN_ID
    if unknown_mask.any():
        warnings.warn(f"[Encoder] {unknown_mask.sum()} tareas con task_name desconocido eliminadas")
        df = df[~unknown_mask]

    # Columnas finales para el modelo
    MODEL_COLS = ["task_id", "day_of_week", "hour", "minute", "duration_mins"]

    weekly_dfs = []
    # Agrupar por (iso_year, iso_week) manteniendo orden cronológico
    for week_key, group in df.groupby("week_key", sort=True):
        week_df = group[MODEL_COLS].reset_index(drop=True)

        if len(week_df) < min_tasks_per_week:
            warnings.warn(
                f"[Semanas] Semana {week_key} descartada "
                f"({len(week_df)} tareas < mínimo {min_tasks_per_week})"
            )
            continue

        # Asegurar tipos correctos
        week_df = week_df.astype({
            "task_id"      : int,
            "day_of_week"  : int,
            "hour"         : int,
            "minute"       : int,
            "duration_mins": float,
        })
        weekly_dfs.append(week_df)

    print(f"[Semanas] {len(weekly_dfs)} semanas válidas  "
          f"| media {np.mean([len(w) for w in weekly_dfs]):.1f} tareas/semana  "
          f"| rango [{min(len(w) for w in weekly_dfs)}, "
          f"{max(len(w) for w in weekly_dfs)}]")
    return weekly_dfs


# ══════════════════════════════════════════════════════════════════════════════
# 4. FUNCIÓN PÚBLICA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def cargar_semanas_desde_json(
    path: Union[str, pathlib.Path],
    encoder_path: Optional[str] = None,
    min_tasks_per_week: int = 10,
    save_encoder: bool = True,
) -> Tuple[List[pd.DataFrame], TaskEncoder]:
    """
    Pipeline completo: JSON  →  lista de DataFrames semanales listos para el modelo.

    Args:
        path               : Ruta al fichero .json con las tareas históricas.
        encoder_path       : Ruta a un vocabulario previo (.json).
                             Si es None, se construye uno nuevo desde los datos.
        min_tasks_per_week : Semanas con menos tareas que este umbral se descartan.
        save_encoder       : Si True y encoder_path es None, guarda el nuevo
                             vocabulario en 'task_encoder.json'.

    Returns:
        weekly_dfs : List[pd.DataFrame]
            Lista ordenada cronológicamente. Cada elemento = una semana.
            Columnas: task_id, day_of_week, hour, minute, duration_mins

        encoder    : TaskEncoder
            Para traducir task_id ↔ task_name en la inferencia final.

    Ejemplo:
        weekly_dfs, encoder = cargar_semanas_desde_json("mis_tareas.json")
        print(f"{len(weekly_dfs)} semanas, {encoder.n_classes} tareas únicas")

        # Usar en el modelo:
        trainer = Trainer(cfg, weekly_dfs)
    """
    print("─" * 55)
    print(" CARGANDO DATOS")
    print("─" * 55)

    # ── Paso 1: Leer y parsear ─────────────────────────────────────────────
    df_raw = _parse_json_file(path)

    # ── Paso 2: Limpiar y extraer features ────────────────────────────────
    df_clean = _clean_and_engineer(df_raw)

    # ── Paso 3: Construir o cargar el codificador de tareas ───────────────
    if encoder_path and pathlib.Path(encoder_path).exists():
        encoder = TaskEncoder.load(encoder_path)
    else:
        encoder = TaskEncoder().fit(df_clean["task_name"])
        print(f"[TaskEncoder] {encoder.n_classes} tareas únicas encontradas:")
        for name, idx in sorted(encoder.name_to_id.items(), key=lambda x: x[1]):
            print(f"  {idx:3d}  {name}")
        if save_encoder:
            encoder.save("task_encoder.json")

    # ── Paso 4: Agrupar en semanas ─────────────────────────────────────────
    weekly_dfs = _group_into_weeks(df_clean, encoder, min_tasks_per_week)

    print("─" * 55)
    print(f" ✓ Listo: {len(weekly_dfs)} semanas  |  {encoder.n_classes} tareas únicas")
    print("─" * 55 + "\n")
    return weekly_dfs, encoder


# ══════════════════════════════════════════════════════════════════════════════
# 5. UTILIDADES DE INFERENCIA  –  Decodificar predicciones
# ══════════════════════════════════════════════════════════════════════════════

def decodificar_predicciones(
    predicted_df: pd.DataFrame,
    encoder: TaskEncoder,
) -> pd.DataFrame:
    """
    Traduce el DataFrame de predicciones (con task_id numérico)
    a un DataFrame legible con el nombre real de cada tarea.

    Args:
        predicted_df : DataFrame devuelto por autoregressive_predict()
        encoder      : TaskEncoder usado en el entrenamiento

    Returns:
        DataFrame con columnas:
          step, task_name, day_of_week, hour, minute, duration_mins
    """
    result = predicted_df.copy()
    result["task_name"] = encoder.inverse_transform(result["task_id"])

    DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    result["day_name"] = result["day_of_week"].map(
        lambda d: DAYS[d] if 0 <= d <= 6 else "?"
    )
    result["time_str"] = result.apply(
        lambda r: f"{int(r['hour']):02d}:{int(r['minute']):02d}", axis=1
    )

    cols = ["step", "task_name", "day_name", "time_str", "duration_mins"]
    return result[cols].rename(columns={
        "day_name"     : "día",
        "time_str"     : "hora",
        "duration_mins": "duración_mins",
    })


# ══════════════════════════════════════════════════════════════════════════════
# 6. BLOQUE PRINCIPAL  –  Test de la función de carga
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Generar un JSON de ejemplo para testear si no se pasa ruta como argumento
    if len(sys.argv) < 2:
        print("[Test] Generando JSON de ejemplo para verificar el cargador...\n")

        import random, json as _json
        from datetime import datetime, timedelta, timezone

        TASK_NAMES = [
            "Fregar suelo entrada", "Limpiar baño principal", "Barrer terraza",
            "Pasar aspiradora salón", "Limpiar cocina", "Fregar platos",
            "Limpiar ventanas", "Sacar basura", "Limpiar microondas",
            "Organizar armario",
        ]
        rng = random.Random(42)
        start_date = datetime(2025, 1, 6, tzinfo=timezone.utc)  # primer lunes
        records = []
        for week in range(52):
            for _ in range(rng.randint(8, 15)):
                day_offset = rng.randint(0, 6)
                hour       = rng.randint(8, 20)
                minute     = rng.choice([0, 15, 30, 45])
                duration   = rng.randint(10, 90)
                start = start_date + timedelta(weeks=week, days=day_offset,
                                               hours=hour, minutes=minute)
                end   = start + timedelta(minutes=duration)
                records.append({
                    "uid"       : f"uid_{week}_{len(records)}",
                    "task_name" : rng.choice(TASK_NAMES),
                    "type"      : "Delivery",
                    "status"    : "Scheduled",
                    "start_time": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "end_time"  : end.strftime(  "%Y-%m-%dT%H:%M:%S.000Z"),
                })
        with open("/tmp/test_tasks.json", "w") as f:
            _json.dump(records, f)
        path = "/tmp/test_tasks.json"
        print(f"JSON de ejemplo creado con {len(records)} tareas\n")
    else:
        path = sys.argv[1]

    # ── Ejecutar la carga ──────────────────────────────────────────────────
    weekly_dfs, encoder = cargar_semanas_desde_json(path)

    # ── Mostrar resumen ────────────────────────────────────────────────────
    print("\nPrimeras 5 tareas de la semana 1:")
    print(weekly_dfs[0].head())
    print(f"\nÚltimas 5 tareas de la semana {len(weekly_dfs)}:")
    print(weekly_dfs[-1].tail())


"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         PREDICTOR DE SECUENCIAS DE TAREAS SEMANALES - PyTorch              ║
║         Arquitectura: Transformer Encoder + Multi-Head Output               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Flujo del sistema:
  [Semanas históricas] → [Dataset con ventana deslizante] → [Transformer]
  → [4 cabezas de salida] → [task_id, day_of_week, hour, minute, duration]

Estructura del fichero:
  1. Configuración e Hiperparámetros       (Config dataclass)
  2. Dataset y DataLoader                  (WeeklyTaskDataset)
  3. Arquitectura del Modelo               (TaskSequenceModel)
  4. Utilidades de Entrenamiento           (Trainer)
  5. Bucle principal Train / Val / Test    (main)
  6. Inferencia Autoregresiva              (autoregressive_predict)
"""

import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 0. SEMILLA GLOBAL  –  Reproducibilidad
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN CENTRAL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # --- Datos ---
    num_task_classes: int = 50          # Número de task_ids únicos (0..49)
    context_weeks:    int = 4           # Cuántas semanas anteriores usa el modelo como contexto
    max_tasks_per_week: int = 120       # Longitud máxima de secuencia (para padding)

    # --- Embeddings ---
    task_embed_dim:   int = 32          # Dimensión embedding task_id
    dow_embed_dim:    int = 8           # Dimensión embedding day_of_week  (0-6)
    hour_embed_dim:   int = 16          # Dimensión embedding hour          (0-23)
    minute_embed_dim: int = 8           # Dimensión embedding minute        (0, 15, 30, 45 → 4 clases)

    # --- Transformer ---
    d_model:          int = 128         # Dimensión interna del Transformer
    nhead:            int = 4           # Número de cabezas de atención
    num_encoder_layers: int = 3         # Capas del encoder
    dim_feedforward:  int = 256         # Dimensión capa feed-forward interna
    dropout:          float = 0.1

    # --- Entrenamiento ---
    epochs:           int = 50
    batch_size:       int = 16
    lr:               float = 3e-4
    weight_decay:     float = 1e-4
    grad_clip:        float = 1.0       # Gradient clipping

    # --- Pesos de la pérdida combinada ---
    # Loss_total = λ_task·L_task + λ_dow·L_dow + λ_hour·L_hour
    #            + λ_min·L_min  + λ_dur·L_dur
    lambda_task:  float = 1.0
    lambda_dow:   float = 0.5
    lambda_hour:  float = 0.5
    lambda_min:   float = 0.3
    lambda_dur:   float = 0.3

    # --- Normalización de variables continuas ---
    # Se calculan sobre el train set (ver Trainer._compute_stats)
    duration_mean: float = 30.0         # Placeholder; se sobreescribe en Trainer
    duration_std:  float = 15.0

    # --- División temporal de semanas ---
    # Con ~52 semanas: 40 train | 6 val | 6 test
    train_end:    int = 40
    val_end:      int = 46              # 40 + 6

    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    ))


CFG = Config()

# ── Vocabularios fijos ──────────────────────────────────────────────────────
# Minuto: discretizamos a cuartos de hora → 4 clases (0,1,2,3) → 0,15,30,45
MINUTE_TO_CLASS = {0: 0, 15: 1, 30: 2, 45: 3}
CLASS_TO_MINUTE = {0: 0, 1: 15, 2: 30, 3: 45}
NUM_MINUTE_CLASSES = 4

DOW_CLASSES  = 7    # 0 = lunes … 6 = domingo
HOUR_CLASSES = 24


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET Y DATALOADER
# ══════════════════════════════════════════════════════════════════════════════

class WeeklyTaskDataset(Dataset):
    """
    Dataset con ventana deslizante sobre semanas cronológicas.

    Entrada que recibe:
        weekly_dfs : List[pd.DataFrame]
            Lista ordenada cronológicamente donde weekly_dfs[i] es el DataFrame
            de la semana i con columnas:
              task_id (int 0-49), day_of_week (int 0-6),
              hour (int 0-23), minute (int), duration_mins (float)

        context_weeks : int
            Número de semanas pasadas que se usan como contexto (entrada).

    Salida de __getitem__  →  (src_tensor, tgt_tensor)
        src_tensor : Tensor [L_src, 5]   — secuencia de contexto aplanada
        tgt_tensor : Tensor [L_tgt, 5]   — semana objetivo a predecir
            Columnas → [task_id, day_of_week, hour, minute_class, duration_norm]

    El DataLoader usará collate_fn para hacer padding y devolver máscaras.
    """

    def __init__(
        self,
        weekly_dfs: List[pd.DataFrame],
        context_weeks: int,
        duration_mean: float = 30.0,
        duration_std: float  = 15.0,
    ):
        self.weekly_dfs    = weekly_dfs
        self.context_weeks = context_weeks
        self.dur_mean      = duration_mean
        self.dur_std       = duration_std

        # El primer índice válido de "semana objetivo" es context_weeks
        # (necesita context_weeks semanas anteriores como contexto)
        self.valid_indices = list(range(context_weeks, len(weekly_dfs)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _df_to_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        """Convierte un DataFrame de semana a tensor [N_tasks, 5]."""
        task_id   = torch.tensor(df["task_id"].values,    dtype=torch.long)
        dow       = torch.tensor(df["day_of_week"].values, dtype=torch.long)
        hour      = torch.tensor(df["hour"].values,        dtype=torch.long)

        # Discretizar minutos a clases
        minute_cls = torch.tensor(
            df["minute"].map(lambda m: MINUTE_TO_CLASS.get(
                int(round(m / 15) * 15) % 60, 0   # redondea al cuarto más cercano
            )).values,
            dtype=torch.long
        )

        # Normalizar duración con z-score (estadísticos del train set)
        dur_norm = torch.tensor(
            (df["duration_mins"].values - self.dur_mean) / (self.dur_std + 1e-8),
            dtype=torch.float32
        )

        # Stack → [N, 5]  (long para ids, float para dur_norm)
        # Guardamos todo como float para poder hacer stack; el modelo
        # hará casting interno según necesite.
        return torch.stack([
            task_id.float(),
            dow.float(),
            hour.float(),
            minute_cls.float(),
            dur_norm
        ], dim=1)   # [N_tasks, 5]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_week_idx = self.valid_indices[idx]

        # ── Contexto: concatenar las context_weeks semanas anteriores ─────────
        context_dfs = self.weekly_dfs[
            target_week_idx - self.context_weeks : target_week_idx
        ]
        src_tensor = torch.cat(
            [self._df_to_tensor(df) for df in context_dfs], dim=0
        )  # [L_src, 5]

        # ── Objetivo: la semana siguiente ─────────────────────────────────────
        tgt_tensor = self._df_to_tensor(self.weekly_dfs[target_week_idx])
        # [L_tgt, 5]

        return src_tensor, tgt_tensor


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Padding de secuencias de longitud variable.
    Devuelve un diccionario con tensores paddeados y máscaras booleanas.
    """
    src_list, tgt_list = zip(*batch)

    # pad_sequence espera lista de tensores [L, features] → [L_max, B, features]
    src_padded = pad_sequence(src_list, batch_first=True, padding_value=0.0)
    tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=0.0)

    # Máscaras: True = posición de padding (ignorar en la loss)
    src_mask = torch.zeros(src_padded.size(0), src_padded.size(1), dtype=torch.bool)
    tgt_mask = torch.zeros(tgt_padded.size(0), tgt_padded.size(1), dtype=torch.bool)

    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src_mask[i, len(s):] = True
        tgt_mask[i, len(t):] = True

    return {
        "src":      src_padded,    # [B, L_src, 5]
        "tgt":      tgt_padded,    # [B, L_tgt, 5]
        "src_mask": src_mask,      # [B, L_src]  — True = padding
        "tgt_mask": tgt_mask,      # [B, L_tgt]
    }


def build_dataloaders(
    weekly_dfs: List[pd.DataFrame],
    cfg: Config,
    dur_mean: float,
    dur_std:  float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    División temporal estricta:
        Train : semanas  0  … cfg.train_end - 1
        Val   : semanas  cfg.train_end … cfg.val_end - 1
        Test  : semanas  cfg.val_end   … fin

    Nota: cada split necesita al menos context_weeks semanas de "warmup",
    por eso pasamos toda la lista pero limitamos valid_indices internamente.
    """
    assert len(weekly_dfs) >= cfg.val_end, (
        f"Se necesitan al menos {cfg.val_end} semanas; tienes {len(weekly_dfs)}"
    )

    # Creamos sub-listas con suficiente contexto histórico para cada split
    # Train: semanas 0 … train_end-1  (válidos desde context_weeks)
    train_weeks = weekly_dfs[:cfg.train_end]
    # Val  : incluye las últimas context_weeks del train para poder generar contexto
    val_weeks   = weekly_dfs[cfg.train_end - cfg.context_weeks : cfg.val_end]
    # Test : ídem
    test_weeks  = weekly_dfs[cfg.val_end - cfg.context_weeks :]

    train_ds = WeeklyTaskDataset(train_weeks, cfg.context_weeks, dur_mean, dur_std)
    val_ds   = WeeklyTaskDataset(val_weeks,   cfg.context_weeks, dur_mean, dur_std)
    test_ds  = WeeklyTaskDataset(test_weeks,  cfg.context_weeks, dur_mean, dur_std)

    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size,
        shuffle=True,  collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    val_dl   = DataLoader(
        val_ds,   batch_size=cfg.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    test_dl  = DataLoader(
        test_ds,  batch_size=cfg.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}")
    print(f"  Test  samples : {len(test_ds)}")
    return train_dl, val_dl, test_dl


# ══════════════════════════════════════════════════════════════════════════════
# 3. ARQUITECTURA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Codificación posicional sinusoidal estándar (Vaswani et al. 2017).
    Añade información de posición en la secuencia al embedding.
    """
    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                    # [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)      # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                   # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model]"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TaskEmbedding(nn.Module):
    """
    Proyecta un paso de secuencia (task_id, dow, hour, minute_cls, dur_norm)
    a un vector denso de dimensión d_model mediante embeddings categóricos
    y una proyección lineal para el valor continuo.

    Diagrama:
      task_id    → Embedding(50,  32)  ┐
      dow        → Embedding(7,    8)  │
      hour       → Embedding(24,  16)  ├── concat → Linear → d_model
      minute_cls → Embedding(4,    8)  │
      dur_norm   → Linear(1, 16)       ┘
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.emb_task   = nn.Embedding(cfg.num_task_classes, cfg.task_embed_dim,   padding_idx=None)
        self.emb_dow    = nn.Embedding(DOW_CLASSES,           cfg.dow_embed_dim)
        self.emb_hour   = nn.Embedding(HOUR_CLASSES,          cfg.hour_embed_dim)
        self.emb_minute = nn.Embedding(NUM_MINUTE_CLASSES,    cfg.minute_embed_dim)
        self.proj_dur   = nn.Linear(1, 16)

        total_dim = (
            cfg.task_embed_dim + cfg.dow_embed_dim +
            cfg.hour_embed_dim + cfg.minute_embed_dim + 16
        )
        self.proj_out = nn.Sequential(
            nn.Linear(total_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, L, 5]
            col 0 → task_id (long)
            col 1 → dow     (long)
            col 2 → hour    (long)
            col 3 → minute_cls (long)
            col 4 → dur_norm   (float)
        """
        task_id    = x[:, :, 0].long()
        dow        = x[:, :, 1].long()
        hour       = x[:, :, 2].long()
        minute_cls = x[:, :, 3].long()
        dur        = x[:, :, 4:5]          # [B, L, 1]

        e_task = self.emb_task(task_id)     # [B, L, 32]
        e_dow  = self.emb_dow(dow)          # [B, L, 8]
        e_hour = self.emb_hour(hour)        # [B, L, 16]
        e_min  = self.emb_minute(minute_cls)# [B, L, 8]
        e_dur  = self.proj_dur(dur)         # [B, L, 16]

        combined = torch.cat([e_task, e_dow, e_hour, e_min, e_dur], dim=-1)
        return self.proj_out(combined)      # [B, L, d_model]


class TaskSequenceModel(nn.Module):
    """
    Modelo Transformer Encoder-Only con múltiples cabezas de salida.

    Arquitectura:
      ┌────────────────────────────────────────────────────────┐
      │  Contexto (src) [B, L_src, 5]                         │
      │        ↓  TaskEmbedding                               │
      │  [B, L_src, d_model]                                  │
      │        ↓  PositionalEncoding                          │
      │        ↓  TransformerEncoder (N capas)                │
      │  encoded [B, L_src, d_model]                          │
      │        ↓  Pooling (mean sobre pasos no-padding)       │
      │  ctx_vec [B, d_model]                                 │
      │        ↓ (se expande y concatena con tgt embedding)  │
      │                                                        │
      │  Objetivo (tgt) [B, L_tgt, 5]                        │
      │        ↓  TaskEmbedding                               │
      │        ↓  PositionalEncoding                          │
      │        ↓  TransformerDecoder simulado (cross-attn)   │
      │        ↓  Cabezas de salida                           │
      │                                                        │
      │  Salidas:                                             │
      │    logits_task  [B, L_tgt, 50]   → CrossEntropy      │
      │    logits_dow   [B, L_tgt, 7]    → CrossEntropy      │
      │    logits_hour  [B, L_tgt, 24]   → CrossEntropy      │
      │    logits_min   [B, L_tgt, 4]    → CrossEntropy      │
      │    pred_dur     [B, L_tgt, 1]    → MSELoss           │
      └────────────────────────────────────────────────────────┘

    Usamos un Transformer encoder para el contexto y un decoder causal para
    generar la secuencia objetivo paso a paso (teacher-forcing en training,
    autoregresión en inferencia).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # ── Embeddings (compartidos src y tgt) ──────────────────────────────
        self.src_embedding = TaskEmbedding(cfg)
        self.tgt_embedding = TaskEmbedding(cfg)

        self.src_pos_enc = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)
        self.tgt_pos_enc = PositionalEncoding(cfg.d_model, dropout=cfg.dropout)

        # ── Encoder  (procesa el contexto histórico) ─────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,    # [B, L, d_model] en todas partes
            norm_first=True      # Pre-LN: más estable en training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_encoder_layers,
            norm=nn.LayerNorm(cfg.d_model)
        )

        # ── Decoder  (genera la secuencia objetivo) ──────────────────────────
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=cfg.num_encoder_layers,
            norm=nn.LayerNorm(cfg.d_model)
        )

        # ── Cabezas de salida (Multi-Head Output) ────────────────────────────
        self.head_task  = nn.Linear(cfg.d_model, cfg.num_task_classes)  # 50 clases
        self.head_dow   = nn.Linear(cfg.d_model, DOW_CLASSES)           #  7 clases
        self.head_hour  = nn.Linear(cfg.d_model, HOUR_CLASSES)          # 24 clases
        self.head_min   = nn.Linear(cfg.d_model, NUM_MINUTE_CLASSES)    #  4 clases
        self.head_dur   = nn.Linear(cfg.d_model, 1)                     # regresión

        self._init_weights()

    def _init_weights(self):
        """Inicialización de pesos con Xavier uniforme."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _make_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """
        Máscara causal (triangular superior) para el decoder:
        cada posición sólo puede atender a posiciones anteriores.
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask  # True = ignorar

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Codifica el contexto histórico.
        src : [B, L_src, 5] → devuelve [B, L_src, d_model]
        """
        x = self.src_embedding(src)           # [B, L_src, d_model]
        x = self.src_pos_enc(x)
        memory = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )
        return memory                          # [B, L_src, d_model]

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decodifica un paso (o secuencia) usando la memoria del encoder.
        tgt : [B, L_tgt, 5] → devuelve [B, L_tgt, d_model]
        """
        L_tgt  = tgt.size(1)
        device = tgt.device

        x = self.tgt_embedding(tgt)            # [B, L_tgt, d_model]
        x = self.tgt_pos_enc(x)

        causal_mask = self._make_causal_mask(L_tgt, device)  # [L_tgt, L_tgt]

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return out                             # [B, L_tgt, d_model]

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Paso forward completo (usado en training con teacher-forcing).

        Retorna:
            {
              "task"  : logits [B, L_tgt, 50],
              "dow"   : logits [B, L_tgt, 7],
              "hour"  : logits [B, L_tgt, 24],
              "min"   : logits [B, L_tgt, 4],
              "dur"   : preds  [B, L_tgt, 1],
            }
        """
        # 1. Encoder
        memory = self.encode(src, src_key_padding_mask)

        # 2. Decoder con teacher-forcing:
        #    La entrada del decoder es la secuencia objetivo desplazada una pos.
        #    (shift right: [BOS] + tgt[:-1]) – simplificamos usando tgt directamente
        #    en la primera posición con máscara causal.
        out = self.decode(tgt, memory, tgt_key_padding_mask, src_key_padding_mask)

        # 3. Proyecciones de salida
        return {
            "task" : self.head_task(out),    # [B, L_tgt, 50]
            "dow"  : self.head_dow(out),     # [B, L_tgt, 7]
            "hour" : self.head_hour(out),    # [B, L_tgt, 24]
            "min"  : self.head_min(out),     # [B, L_tgt, 4]
            "dur"  : self.head_dur(out),     # [B, L_tgt, 1]
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. PÉRDIDA COMBINADA
# ══════════════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    Loss total = λ_task·CE(task) + λ_dow·CE(dow)  + λ_hour·CE(hour)
               + λ_min·CE(min)  + λ_dur·MSE(dur)

    Las posiciones de padding se excluyen usando la tgt_mask.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # reduction='none' para aplicar la máscara manualmente
        self.ce  = nn.CrossEntropyLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")

    def _masked_mean(
        self,
        loss_per_token: torch.Tensor,   # [B, L]
        mask: torch.Tensor              # [B, L] True = padding
    ) -> torch.Tensor:
        """Calcula la media ignorando tokens de padding."""
        valid = (~mask).float()                  # [B, L] 1 = válido
        return (loss_per_token * valid).sum() / valid.sum().clamp(min=1)

    def forward(
        self,
        preds:    Dict[str, torch.Tensor],
        targets:  torch.Tensor,           # [B, L_tgt, 5]
        tgt_mask: torch.Tensor            # [B, L_tgt] True = padding
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Retorna (loss_total, dict_de_métricas).
        """
        B, L, _ = targets.shape

        task_tgt = targets[:, :, 0].long()   # [B, L]
        dow_tgt  = targets[:, :, 1].long()
        hour_tgt = targets[:, :, 2].long()
        min_tgt  = targets[:, :, 3].long()
        dur_tgt  = targets[:, :, 4:5]        # [B, L, 1]

        # ── Clasificaciones ──────────────────────────────────────────────────
        # CrossEntropyLoss espera [B*L, C] y [B*L]
        def ce_masked(logits, tgt):
            l = self.ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            return self._masked_mean(l.view(B, L), tgt_mask)

        l_task = ce_masked(preds["task"], task_tgt)
        l_dow  = ce_masked(preds["dow"],  dow_tgt)
        l_hour = ce_masked(preds["hour"], hour_tgt)
        l_min  = ce_masked(preds["min"],  min_tgt)

        # ── Regresión duración ───────────────────────────────────────────────
        l_dur  = self._masked_mean(
            self.mse(preds["dur"], dur_tgt).squeeze(-1),   # [B, L]
            tgt_mask
        )

        cfg = self.cfg
        total = (
            cfg.lambda_task  * l_task +
            cfg.lambda_dow   * l_dow  +
            cfg.lambda_hour  * l_hour +
            cfg.lambda_min   * l_min  +
            cfg.lambda_dur   * l_dur
        )

        metrics = {
            "total": total.item(),
            "task":  l_task.item(),
            "dow":   l_dow.item(),
            "hour":  l_hour.item(),
            "min":   l_min.item(),
            "dur":   l_dur.item(),
        }
        return total, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINER  –  Bucle de Entrenamiento y Evaluación
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Encapsula el ciclo de entrenamiento, validación y evaluación.

    Responsabilidades:
      · Calcular estadísticas de normalización sobre el train set.
      · Construir DataLoaders con división temporal correcta.
      · Ejecutar epochs de train / val con logging de métricas.
      · Early stopping basado en val_loss.
      · Guardar y cargar el mejor modelo (checkpoint).
    """

    def __init__(self, cfg: Config, weekly_dfs: List[pd.DataFrame]):
        self.cfg    = cfg
        self.device = torch.device(cfg.device)

        # ── Estadísticas de normalización (sólo del train set) ────────────────
        train_dfs = weekly_dfs[:cfg.train_end]
        dur_mean, dur_std = self._compute_stats(train_dfs)
        cfg.duration_mean = dur_mean
        cfg.duration_std  = dur_std
        print(f"[Normalización] duration: μ={dur_mean:.2f}  σ={dur_std:.2f}")

        # ── DataLoaders ───────────────────────────────────────────────────────
        print("[DataLoaders]")
        self.train_dl, self.val_dl, self.test_dl = build_dataloaders(
            weekly_dfs, cfg, dur_mean, dur_std
        )

        # ── Modelo, optimizador y scheduler ──────────────────────────────────
        self.model     = TaskSequenceModel(cfg).to(self.device)
        self.criterion = CombinedLoss(cfg)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
        # Cosine Annealing: reduce lr suavemente hasta 0 a lo largo de los epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[Modelo] Parámetros totales: {total_params:,}")

    @staticmethod
    def _compute_stats(dfs: List[pd.DataFrame]) -> Tuple[float, float]:
        """Calcula media y desviación estándar de duration_mins."""
        all_dur = pd.concat(dfs)["duration_mins"].values.astype(float)
        return float(all_dur.mean()), float(all_dur.std())

    def _batch_to_device(self, batch: Dict) -> Dict:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _run_epoch(
        self,
        dl: DataLoader,
        train: bool = True
    ) -> Dict[str, float]:
        """Ejecuta un epoch completo (train o eval)."""
        self.model.train(train)
        accum = {k: 0.0 for k in ["total","task","dow","hour","min","dur"]}
        n_batches = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in dl:
                batch = self._batch_to_device(batch)
                src      = batch["src"]       # [B, L_src, 5]
                tgt      = batch["tgt"]       # [B, L_tgt, 5]
                src_mask = batch["src_mask"]  # [B, L_src]
                tgt_mask = batch["tgt_mask"]  # [B, L_tgt]

                # ── Forward ─────────────────────────────────────────────────
                preds = self.model(
                    src, tgt,
                    src_key_padding_mask=src_mask,
                    tgt_key_padding_mask=tgt_mask,
                )
                loss, metrics = self.criterion(preds, tgt, tgt_mask)

                # ── Backward ─────────────────────────────────────────────────
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                    self.optimizer.step()

                for k, v in metrics.items():
                    accum[k] += v
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    def train(self) -> None:
        """Bucle principal de entrenamiento con early stopping."""
        best_val_loss = float("inf")
        patience      = 10        # epochs sin mejora antes de detener
        patience_cnt  = 0
        best_state    = None

        print("\n" + "═" * 60)
        print("  ENTRENAMIENTO")
        print("═" * 60)

        for epoch in range(1, self.cfg.epochs + 1):
            train_m = self._run_epoch(self.train_dl, train=True)
            val_m   = self._run_epoch(self.val_dl,   train=False)
            self.scheduler.step()

            lr_now = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{self.cfg.epochs}  "
                f"│ Train {train_m['total']:.4f} "
                f"(task={train_m['task']:.3f} dow={train_m['dow']:.3f} "
                f"hour={train_m['hour']:.3f} dur={train_m['dur']:.3f}) "
                f"│ Val {val_m['total']:.4f}  "
                f"│ lr={lr_now:.2e}"
            )

            # ── Checkpoint del mejor modelo ──────────────────────────────────
            if val_m["total"] < best_val_loss:
                best_val_loss = val_m["total"]
                best_state    = {k: v.cpu().clone()
                                 for k, v in self.model.state_dict().items()}
                patience_cnt  = 0
                torch.save(best_state, "best_model.pt")
                print(f"           ✓ Nuevo mejor modelo guardado (val={best_val_loss:.4f})")
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"\nEarly stopping en epoch {epoch}")
                    break

        # Restaurar mejor estado
        if best_state is not None:
            self.model.load_state_dict(best_state)
        print("\n[Entrenamiento completado]")

    def evaluate_test(self) -> Dict[str, float]:
        """Evalúa el modelo en el test set."""
        print("\n[Evaluación en Test Set]")
        test_m = self._run_epoch(self.test_dl, train=False)
        for k, v in test_m.items():
            print(f"  {k:8s}: {v:.4f}")
        return test_m
    
    


# ══════════════════════════════════════════════════════════════════════════════
# 6. INFERENCIA AUTOREGRESIVA
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def autoregressive_predict(
    model: TaskSequenceModel,
    context_weeks: List[pd.DataFrame],
    cfg: Config,
    num_steps: int = 100,
    temperature: float = 1.0,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Genera autoregresivamente la secuencia de tareas para la próxima semana.

    Algoritmo:
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. Codificar semanas de contexto  →  memory [1, L_src, d_model]│
    │  2. Inicializar tgt con token BOS (todo ceros)                  │
    │  3. Para cada paso t = 1 … num_steps:                           │
    │       a. Decodificar tgt[:t]  →  out[:, -1, :]                 │
    │       b. Aplicar cada cabeza  →  logits / pred                  │
    │       c. Samplear / argmax    →  tokens nuevos                  │
    │       d. Añadir tokens a tgt  →  tgt[:t+1]                     │
    │  4. Desnormalizar duración y devolver DataFrame                 │
    └─────────────────────────────────────────────────────────────────┘

    Args:
        model         : modelo entrenado en eval mode
        context_weeks : list de cfg.context_weeks DataFrames recientes
        cfg           : Config con parámetros de normalización
        num_steps     : tareas a predecir (~100 para una semana)
        temperature   : >1 → más diversidad; <1 → más determinismo
        top_k         : muestrea sólo entre los top_k tokens más probables

    Returns:
        pd.DataFrame con columnas:
          [task_id, day_of_week, hour, minute, duration_mins]
    """
    device = torch.device(cfg.device)
    model  = model.to(device).eval()

    # ── Construir el tensor de contexto ──────────────────────────────────────
    ds_tmp = WeeklyTaskDataset(
        context_weeks, len(context_weeks),
        cfg.duration_mean, cfg.duration_std
    )
    src_list = [ds_tmp._df_to_tensor(df) for df in context_weeks]
    src = torch.cat(src_list, dim=0).unsqueeze(0).to(device)  # [1, L_src, 5]

    # ── Codificar el contexto (una sola vez) ─────────────────────────────────
    memory = model.encode(src)  # [1, L_src, d_model]

    # ── Token BOS: vector de ceros (representa "inicio de secuencia") ─────────
    bos = torch.zeros(1, 1, 5, dtype=torch.float32, device=device)  # [1, 1, 5]
    generated_tokens = bos.clone()  # [1, t, 5]  (crecerá en cada paso)

    predictions = []  # lista de dicts

    def sample_logits(logits_1d: torch.Tensor) -> int:
        """Top-k sampling con temperatura."""
        logits_1d = logits_1d / temperature
        if top_k > 0:
            vals, _ = torch.topk(logits_1d, min(top_k, logits_1d.size(-1)))
            logits_1d[logits_1d < vals[-1]] = -float("inf")
        probs = torch.softmax(logits_1d, dim=-1)
        return torch.multinomial(probs, 1).item()

    for step in range(num_steps):
        # Decodificar toda la secuencia generada hasta ahora
        out = model.decode(generated_tokens, memory)  # [1, t+1, d_model]

        # Tomar sólo el ÚLTIMO token generado
        last = out[:, -1, :]  # [1, d_model]

        # ── Proyecciones ──────────────────────────────────────────────────────
        logits_task = model.head_task(last).squeeze(0)  # [50]
        logits_dow  = model.head_dow(last).squeeze(0)   # [7]
        logits_hour = model.head_hour(last).squeeze(0)  # [24]
        logits_min  = model.head_min(last).squeeze(0)   # [4]
        pred_dur    = model.head_dur(last).squeeze()     # escalar

        # ── Muestreo / predicción ──────────────────────────────────────────────
        task_id    = sample_logits(logits_task)
        dow        = sample_logits(logits_dow)
        hour       = sample_logits(logits_hour)
        min_cls    = sample_logits(logits_min)
        minute     = CLASS_TO_MINUTE[min_cls]

        # Desnormalizar duración
        dur_real = pred_dur.item() * cfg.duration_std + cfg.duration_mean
        dur_real = max(1.0, dur_real)  # al menos 1 minuto

        predictions.append({
            "step"         : step + 1,
            "task_id"      : task_id,
            "day_of_week"  : dow,
            "hour"         : hour,
            "minute"       : minute,
            "duration_mins": round(dur_real, 1),
        })

        # ── Construir nuevo token y añadirlo a la secuencia ──────────────────
        new_token = torch.tensor(
            [[task_id, dow, hour, min_cls,
              (dur_real - cfg.duration_mean) / (cfg.duration_std + 1e-8)]],
            dtype=torch.float32, device=device
        ).unsqueeze(0)  # [1, 1, 5]
        generated_tokens = torch.cat([generated_tokens, new_token], dim=1)

    result_df = pd.DataFrame(predictions)
    return result_df




# ══════════════════════════════════════════════════════════════════════════════
# 7. FUNCIÓN PRINCIPAL  –  Ejemplo de Uso Completo
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(
    n_weeks: int = 52,
    tasks_per_week: int = 100,
    n_task_names: int = 50,
    seed: int = 42,
) -> List[pd.DataFrame]:
    """
    Genera datos sintéticos para demostrar el pipeline.
    REEMPLAZA ESTA FUNCIÓN con tu cargador de datos real.

    Retorna una lista de DataFrames cronológicamente ordenados,
    uno por semana, con columnas:
      task_id, day_of_week, hour, minute, duration_mins
    """
    rng = np.random.default_rng(seed)
    weeks = []
    for week_idx in range(n_weeks):
        n = rng.integers(80, tasks_per_week + 1)  # variabilidad en nº de tareas
        df = pd.DataFrame({
            "task_id"     : rng.integers(0, n_task_names, size=n),
            "day_of_week" : rng.integers(0, 7,            size=n),
            "hour"        : rng.integers(7, 22,           size=n),
            "minute"      : rng.choice([0, 15, 30, 45],   size=n),
            "duration_mins": rng.normal(30, 15, size=n).clip(5, 120),
        })
        weeks.append(df)
    return weeks


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Task Sequence Predictor – PyTorch Transformer          ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    cfg = Config()

    # ── 1. Cargar datos ────────────────────────────────────────────────────────
    print("[1/4] Cargando datos...")
    weekly_dfs, encoder = cargar_semanas_desde_json("aux_databse.json")
    print(f"     {len(weekly_dfs)} semanas cargadas, "
          f"{sum(len(d) for d in weekly_dfs)} tareas totales")

    # ⚠️  Actualizar num_task_classes con el vocabulario REAL del encoder
    cfg.num_task_classes = encoder.n_classes

    # ── 2. Crear Trainer (inicializa modelo + dataloaders) ─────────────────────
    print("\n[2/4] Inicializando Trainer...")
    trainer = Trainer(cfg, weekly_dfs)

    # ── 3. Entrenar ───────────────────────────────────────────────────────────
    print("\n[3/4] Entrenando...")
    trainer.train()

    # ── 4. Evaluar en test ────────────────────────────────────────────────────
    print("\n[4/4] Evaluación en Test Set...")
    trainer.evaluate_test()

    # ── 5. Inferencia autoregresiva ───────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  INFERENCIA AUTOREGRESIVA")
    print("═" * 60)
    print("Generando secuencia de ~100 tareas para la próxima semana...")

    context = weekly_dfs[-cfg.context_weeks:]

    predicted_week = autoregressive_predict(
        model         = trainer.model,
        context_weeks = context,
        cfg           = cfg,
        num_steps     = 100,
        temperature   = 0.8,
        top_k         = 5,
    )

    # ── 6. Decodificar y mostrar resultados legibles ──────────────────────────
    predicciones_legibles = decodificar_predicciones(predicted_week, encoder)
    print(f"\n✓ {len(predicciones_legibles)} tareas generadas:")
    print(predicciones_legibles.head(20).to_string(index=False))
    print(f"  ... ({len(predicciones_legibles) - 20} más)")

    # Guardar ambas versiones
    predicted_week.to_csv("predicted_week_raw.csv", index=False)
    predicciones_legibles.to_csv("predicted_week.csv", index=False)
    print("\n✓ Predicciones guardadas en 'predicted_week.csv'")

    return trainer, predicciones_legibles


if __name__ == "__main__":
    trainer, predictions = main()