from pathlib import Path

# ── Rutas ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "aux_database_reduced.json"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# ── Datos ──────────────────────────────────────────────────────────────────────
TIMEZONE = None              # Ej.: "Europe/Madrid" si tus timestamps están en UTC y quieres convertirlos
WINDOW_WEEKS = 8             # semanas de contexto
BIN_MINUTES = 5              # granularidad temporal
TRAIN_RATIO = 0.80           # split cronológico
SEED = 42

# ── Límites del problema ───────────────────────────────────────────────────────
MAX_OCCURRENCES_PER_TASK = 10   # máximo de repeticiones semanales por tarea
MAX_TASKS_PER_WEEK = 64         # solo para validación/sanidad, no para padding secuencial

# ── Modelo de ocurrencias ──────────────────────────────────────────────────────
OCC_HIDDEN_SIZE = 128
OCC_NUM_LAYERS = 2
OCC_DROPOUT = 0.15
OCC_BATCH_SIZE = 32
OCC_LR = 1e-3
OCC_WEIGHT_DECAY = 1e-4
OCC_MAX_EPOCHS = 250
OCC_PATIENCE = 30

# ── Modelo temporal ────────────────────────────────────────────────────────────
TMP_HIDDEN_SIZE = 128
TMP_NUM_LAYERS = 2
TMP_DROPOUT = 0.15
TASK_EMBED_DIM = 32
OCC_EMBED_DIM = 16
TMP_BATCH_SIZE = 128
TMP_LR = 1e-3
TMP_WEIGHT_DECAY = 1e-4
TMP_MAX_EPOCHS = 250
TMP_PATIENCE = 30
DURATION_LOSS_WEIGHT = 0.25

# ── Ingeniería de características ───────────────────────────────────────────────
HISTORY_SCALES = (1, 2, 4, 8)
START_TOPK_TOLERANCE_BINS = 6   # ±30 min con bin de 5 min

# ── Misc ───────────────────────────────────────────────────────────────────────
DEVICE = "cuda"  # se resuelve dinámicamente a cpu si no hay GPU
VERBOSE_EVERY = 5


def num_time_bins() -> int:
    return (7 * 24 * 60) // BIN_MINUTES
