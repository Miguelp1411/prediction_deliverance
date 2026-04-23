# v4.7: Optuna + gating + selección automática de etapa final

## Qué añade esta versión

- **Etapa final adaptativa (`gated`)**
  - El pipeline ya no obliga a usar siempre `raw`, `anchors` o `rerank`.
  - Para cada semana, calcula `raw`, `anchors` y `rerank`, estima:
    - confianza del modelo (`top1 - top2`)
    - conflictos/solapes
    - movimiento respecto a `raw`
    - acuerdo con anchors
  - Con eso decide automáticamente qué etapa usar.

- **Selección automática de la mejor etapa en validación**
  - La ablación incluye ahora: `raw`, `anchors`, `rerank`, `gated`.
  - Se calcula un score objetivo por etapa y se guarda:
    - `selected_final_stage_validation`
    - `validation_stage_scores`
  - También aparece en `training_report.json`.

- **Optuna para tuning de postproceso/gating**
  - Nuevo script: `tune_v47_optuna.py`
  - Optimiza parámetros de:
    - anchors
    - rerank
    - gating
  - Guarda los mejores parámetros en:
    - `reports/best_postprocess_overrides.json`

- **Carga automática de overrides**
  - `train.py` y `predict.py` intentan cargar el JSON de overrides si existe.

## Uso recomendado

### 1) Entrenar normal
```bash
python3 train.py
```

### 2) Afinar postproceso con Optuna
```bash
python3 tune_v47_optuna.py --trials 32 --objective-mode gated
```

También puedes optimizar por la mejor etapa global de validación:
```bash
python3 tune_v47_optuna.py --trials 32 --objective-mode validation_best
```

### 3) Volver a entrenar o evaluar con overrides ya aplicados
```bash
python3 train.py
python3 predict.py --stage final
```

## Modos útiles

```bash
python3 predict.py --stage raw
python3 predict.py --stage anchors
python3 predict.py --stage rerank
python3 predict.py --stage gated
python3 predict.py --stage final
```

`final` usa lo que marque `TEMPORAL_FINAL_STAGE` en `config.py`.
En v4.7 queda por defecto en `gated`.
