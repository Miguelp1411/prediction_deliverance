Cambios aplicados — activación automática del corrector auxiliar
=============================================================

Objetivo
--------
Que el corrector auxiliar solo se use en predicción si durante el entrenamiento/validación demuestra mejora real.

Qué se añadió
-------------

1. Policy automática guardada en `checkpoints/auxiliary_policy.json`
   - `use_auxiliary`
   - `use_aux_count`
   - `use_aux_duration`
   - `use_aux_time`
   - métricas/auditoría de selección

2. Selección automática en `train.py`
   - entrena el auxiliar sobre train
   - evalúa en validación:
     - base
     - candidato count
     - candidato duration
     - candidato time
   - hace selección progresiva y conservadora
   - solo activa cada submódulo si mejora según reglas de aceptación

3. Consumo automático de policy en `predict.py`
   - carga `auxiliary_policy.json` si existe
   - si la policy desactiva el auxiliar, no lo aplica
   - si la policy activa solo algunos submódulos, aplica solo esos
   - flags CLI siguen pudiendo forzar/inhabilitar módulos

Archivos modificados
--------------------
- `auxiliary_corrector.py`
- `train.py`
- `predict.py`

Notas de uso
------------
1. Ejecuta primero `train.py` para regenerar:
   - `checkpoints/auxiliary_corrector.pt`
   - `checkpoints/auxiliary_policy.json`

2. Después `predict.py` usará la policy automáticamente.

3. Overrides útiles:
   - `--ignore-aux-policy`
   - `--enable-aux-count`
   - `--enable-aux-duration`
   - `--enable-aux-time`
   - `--disable-aux-count`
   - `--disable-aux-duration`
   - `--disable-aux-time`
