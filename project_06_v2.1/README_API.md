# README — Sistema de predicciones del proyecto

## 1. Objetivo

Este documento explica **únicamente la parte de predicciones** del proyecto y cubre:

- el flujo completo de información **desde el frontend hasta el backend**;
- la **API implicada en predicción** y cómo funciona;
- los **archivos que intervienen** en la generación de predicciones;
- **dónde se guardan** los resultados cuando se ejecuta una predicción;
- los **ficheros concretos** donde aparece cada pieza de lógica;
- varias **observaciones importantes** detectadas en la implementación actual.

---

## 2. Visión general

El sistema de predicción está montado sobre una arquitectura de dos capas:

1. **Frontend (React + Vite + Chakra UI)**
   - Muestra un panel lateral de “Predicción” dentro del **Real Mode**.
   - El usuario indica cuántas semanas quiere predecir.
   - El frontend lanza una petición `POST` al backend.

2. **Backend (FastAPI + PyTorch)**
   - Recibe la petición autenticada.
   - Carga el histórico base desde un JSON.
   - Usa dos modelos ya entrenados:
     - **OccurrenceModel**: decide cuántas veces aparecerá cada tarea en la siguiente semana.
     - **TemporalAssignmentModel**: asigna día, hora y duración a cada ocurrencia.
   - Guarda dos salidas:
     - una con **solo las tareas predichas**;
     - otra con el **histórico original + la predicción añadida**.

---

## 3. Flujo completo de información: del front al back

### 3.1. Dónde nace la interacción en el frontend

La UI de predicción está integrada en la pantalla principal del sitio, dentro del modo real:

- `src/routes/_layout/site.tsx`
  - controla si el sistema está en **Simulation** o **Real Mode**;
  - controla cuál de los paneles laterales está abierto;
  - monta `PredictionPanel` solo cuando `isRealMode === true`.

### 3.2. Secuencia funcional

#### Paso 1. El usuario entra en Real Mode
En `site.tsx` se usa este estado para activar el comportamiento del modo real:

- `isRealMode`
- `openRealPanel`

Cuando cambia el modo real, el código resetea el panel abierto a `null`, por lo que los paneles arrancan cerrados.

#### Paso 2. El usuario abre el panel de predicción
`PredictionPanel` recibe estas props:

- `isVisible`
- `isOpen`
- `onToggle`

Eso permite que el panel:
- solo exista visualmente en modo real;
- pueda abrirse o cerrarse desde `site.tsx`;
- comparta lógica con el panel de robots.

#### Paso 3. El usuario indica el número de semanas
En `src/components/Blueprint3D/PredictionPanel.tsx` hay un `NumberInput` que permite elegir `weeks` entre `1` y `52`.

#### Paso 4. El frontend obtiene el token
Cuando se pulsa **Predecir**, `PredictionPanel.tsx` lee el token desde:

```ts
localStorage.getItem("access_token")
```

Ese token se envía en la cabecera `Authorization: Bearer ...`.

#### Paso 5. El frontend llama a la API
La petición que realmente se lanza es:

```http
POST /api/v1/prediction/predict_week
Content-Type: application/json
Authorization: Bearer <token>
```

Body:

```json
{
  "weeks_ahead": 1
}
```

#### Paso 6. El backend autentica al usuario
La ruta de predicción depende de `CurrentUser`, que a su vez valida el JWT con `OAuth2PasswordBearer`.

Eso significa que **no se puede predecir sin login válido**.

#### Paso 7. El backend carga modelos y datos
La ruta de backend:

- lee el histórico base desde `aux_database_1.json`;
- carga los checkpoints:
  - `occurrence_model.pt`
  - `temporal_model.pt`
- convierte el histórico a `DataFrame`;
- prepara las semanas y las features necesarias.

#### Paso 8. Se genera la predicción
Para cada semana pedida:

1. se prepara el histórico actual (`working_records`);
2. el modelo de ocurrencias predice cuántas veces aparece cada tarea;
3. el modelo temporal asigna día, hora y duración;
4. el backend materializa los timestamps finales en UTC;
5. la semana predicha se añade a:
   - `predicted_records` (salida final);
   - `working_records` (para poder predecir la semana siguiente si `weeks_ahead > 1`).

Esto implica que, cuando pides varias semanas, la predicción es **autoregresiva/recursiva**:

- la semana 2 usa como entrada también la semana 1 predicha;
- la semana 3 usa histórico real + semana 1 predicha + semana 2 predicha;
- etc.

#### Paso 9. El backend guarda el resultado
Al terminar, el backend escribe dos archivos:

- `app/api/prediction/predicted_tasks.json`
  - contiene **solo** las tareas predichas en esa ejecución.

- `app/api/prediction/predicted_database.json`
  - contiene el **histórico base + las semanas predichas añadidas**.

#### Paso 10. El frontend muestra el resultado
El frontend no dibuja todavía la predicción dentro de la escena. Ahora mismo, tras la petición:

- muestra un texto de confirmación;
- informa del número de tareas generadas;
- muestra el nombre del archivo combinado (`predicted_database.json`).

Es decir: **la predicción se genera y se guarda, pero no parece consumirse visualmente en el front en esta parte del código**.

---

## 4. Cómo se resuelve la conexión frontend ↔ backend

### En desarrollo
En desarrollo, el frontend usa Vite con proxy:

- `src/main.tsx` pone `OpenAPI.BASE = ""` en desarrollo.
- `vite.config.ts` redirige `/api` y `/static` a `VITE_API_URL`.
- `.env` define:

```env
VITE_API_URL=http://host.docker.internal:8000
```

Por tanto, cuando el frontend llama a:

```ts
fetch("/api/v1/prediction/predict_week", ...)
```

Vite hace de proxy hacia el backend real.

### En producción
En producción, `OpenAPI.BASE` usa `VITE_API_URL` o `http://127.0.0.1:8000`.

---

## 5. API de predicción y cómo funciona

## 5.1. Router principal
La API de predicción se registra en:

- `app/api/main.py`

Ahí se incluye:

```python
api_router.include_router(prediction.router, prefix="/prediction", tags=["prediction"])
```

Y en `app/main.py` todo el router se expone con el prefijo general:

```python
app.include_router(api_router, prefix="/api/v1")
```

Por tanto, la ruta final es:

```text
/api/v1/prediction/...
```

---

## 5.2. Endpoint `POST /api/v1/prediction/predict_week`

### Qué hace
Genera una o varias semanas futuras usando el histórico base y los modelos ya entrenados.

### Request
```json
{
  "weeks_ahead": 1
}
```

### Restricciones
- mínimo: `1`
- máximo: `52`

### Requiere autenticación
Sí. La firma de la ruta incluye `current_user: CurrentUser`.

### Flujo interno
1. Obtiene modelos con `get_models()`.
2. Lee `aux_database_1.json`.
3. Crea:
   - `working_records`: copia del histórico base;
   - `predicted_records`: acumulador solo de tareas predichas.
4. Para cada semana futura:
   - transforma `working_records` a DataFrame;
   - ejecuta `prepare_data(...)`;
   - llama a `predict_next_week(...)`;
   - materializa timestamps finales;
   - añade la predicción a los acumuladores.
5. Escribe:
   - `predicted_tasks.json`
   - `predicted_database.json`
6. Devuelve JSON al frontend.

### Response real
La respuesta tiene esta forma general:

```json
{
  "ok": true,
  "weeks_ahead": 2,
  "generated_count": 95,
  "predicted_file": "predicted_tasks.json",
  "combined_file": "predicted_database.json",
  "data": [
    {
      "uid": null,
      "device_uid": null,
      "task_name": null,
      "type": "Patroll",
      "status": null,
      "start_time": "2028-01-03T08:00:00.000Z",
      "end_time": "2028-01-03T08:24:20.000Z",
      "mileage": 0,
      "misc": null,
      "waypoints": []
    }
  ]
}
```

### Significado de los campos
- `ok`: indica éxito.
- `weeks_ahead`: número de semanas pedidas.
- `generated_count`: número total de tareas generadas.
- `predicted_file`: fichero de solo predicción.
- `combined_file`: fichero histórico + predicción.
- `data`: lista de tareas materializadas.

---

## 5.3. Endpoint `GET /api/v1/prediction/predicted_tasks`

### Qué hace
Devuelve el contenido de `predicted_tasks.json`.

### Comportamiento
- si el archivo no existe, devuelve `[]`;
- si existe, devuelve la lista completa.

### Uso real en el frontend
En el código revisado **no aparece ningún consumo activo** de este endpoint desde el front.

---

## 5.4. Endpoint de login necesario para predicción

Aunque no pertenece al módulo de predicción, forma parte del flujo real:

### `POST /api/v1/login/access-token`
Devuelve el JWT que luego el frontend guarda en `localStorage` como `access_token`.

Sin ese token, la llamada a predicción fallará por autenticación.

---

## 6. Qué archivos usa la predicción

## 6.1. Datos de entrada base
### `app/api/prediction/aux_database_1.json`
Es el **histórico principal** que usa el endpoint de predicción.

Es la fuente real desde la que arranca el backend cuando generas nuevas semanas.

### Formato esperado
Cada registro necesita, como mínimo:

- `type` o `task_name`
- `start_time`
- `end_time`

En este proyecto, el histórico principal usa sobre todo `type`, por ejemplo:

```json
{
  "uid": "...",
  "device_uid": "...",
  "task_name": null,
  "type": "Delivery",
  "status": "Scheduled",
  "start_time": "2026-01-05T08:25:00.000Z",
  "end_time": "2026-01-05T08:35:00.000Z",
  "mileage": 0,
  "misc": null,
  "waypoints": []
}
```

---

## 6.2. Modelos entrenados
### `app/api/prediction/checkpoints/occurrence_model.pt`
Checkpoint del modelo que predice **cuántas veces aparece cada tarea** en la siguiente semana.

### `app/api/prediction/checkpoints/temporal_model.pt`
Checkpoint del modelo que predice **día, hora y duración** para cada ocurrencia.

---

## 6.3. Archivos de configuración
### `app/api/prediction/config.py`
Aquí se definen parámetros clave, por ejemplo:

- `DATA_PATH`
- `CHECKPOINT_DIR`
- `WINDOW_WEEKS = 16`
- `BIN_MINUTES = 5`
- `TRAIN_RATIO = 0.80`
- `PREDICTION_DAY_TOPK = 3`
- `PREDICTION_TIME_TOPK = 8`
- `PREDICTION_REPAIR_RADIUS_BINS = 72`
- `PREDICTION_USE_DURATION_MEDIAN_BLEND = 0.35`

Estos parámetros condicionan cómo se construye el input temporal y cómo se “reparan” conflictos entre tareas.

---

## 7. Qué archivos genera y dónde los guarda

## 7.1. `predicted_tasks.json`
Ruta:

```text
app/api/prediction/predicted_tasks.json
```

Contenido:
- solo las tareas recién predichas;
- formato materializado con fechas `start_time` y `end_time` en UTC.

Uso:
- sirve para consultar solo la salida de la última ejecución.

---

## 7.2. `predicted_database.json`
Ruta:

```text
app/api/prediction/predicted_database.json
```

Contenido:
- el histórico original (`aux_database_1.json`) **más** la predicción generada.

Uso:
- sirve como base combinada para inspección, exportación o futuras integraciones.

---

## 7.3. Importante sobre el guardado
Cada nueva ejecución del endpoint **sobrescribe** ambos archivos de salida.

Es decir:
- no se versionan automáticamente;
- no se guarda historial de ejecuciones;
- siempre se conserva solo el último resultado escrito.

---

## 8. Cómo funciona internamente la predicción

## 8.1. Carga y normalización de datos
Archivo principal:

- `app/api/prediction/data/io.py`

Responsabilidades:
- leer JSON desde disco o desde una lista de registros;
- escoger `type` o `task_name` como etiqueta válida;
- convertir `start_time` y `end_time` a `datetime` en UTC;
- calcular `duration_minutes`.

---

## 8.2. Preprocesado semanal
Archivo principal:

- `app/api/prediction/data/preprocessing.py`

Responsabilidades:
- agrupar tareas por semanas;
- construir `WeekRecord` con estadísticas por tarea;
- convertir cada semana a un vector con:
  - conteos;
  - medias temporales cíclicas;
  - duración media normalizada;
  - features de calendario;
- construir `TemporalContext` por tarea/ocurrencia con:
  - historial reciente;
  - anchor temporal;
  - offsets;
  - recurrencia;
  - señales estacionales.

### Idea clave
El sistema no predice directamente timestamps absolutos “desde cero”.
Primero aprende patrones semanales y luego usa un contexto temporal enriquecido para colocar cada ocurrencia.

---

## 8.3. Modelo 1: ocurrencias
Archivo:

- `app/api/prediction/models/occurrence_model.py`

Qué hace:
- recibe una secuencia de semanas históricas;
- usa un encoder GRU bidireccional con atención;
- produce, para cada tarea, una distribución sobre el número de ocurrencias de la semana siguiente.

Resultado:
- por cada tarea obtiene un número entero `0..max_count_cap`.

---

## 8.4. Modelo 2: asignación temporal
Archivo:

- `app/api/prediction/models/temporal_model.py`

Qué hace:
- usa el contexto secuencial semanal;
- incorpora embeddings de:
  - `task_id`
  - `occurrence_index`
  - `anchor_day`
- mezcla además `history_features` y `predicted_count_norm`;
- predice:
  - `day_logits` → día de la semana;
  - `time_logits` → franja horaria dentro del día;
  - `pred_duration_norm` → duración normalizada.

Resultado:
- para cada ocurrencia se obtiene una propuesta de día, hora y duración.

---

## 8.5. Reparación de solapes
Archivo:

- `app/api/prediction/predict.py`

Una vez el modelo genera candidatos temporales:

- se construyen combinaciones Top-K de día y hora;
- se selecciona la mejor candidata posible;
- si hay conflicto temporal entre tareas, `_repair_predictions(...)` busca un hueco cercano válido.

Esto evita que dos tareas queden solapadas en la semana predicha.

---

## 8.6. Mezcla de duración aprendida + mediana histórica
La duración final no sale solo del modelo.

Se combina:
- la duración predicha por red neuronal;
- la mediana histórica de duración de esa tarea.

Controlado por:

```python
PREDICTION_USE_DURATION_MEDIAN_BLEND = 0.35
```

Eso aporta más estabilidad a la salida.

---

## 9. Diferencia entre los dos JSON de salida

## `predicted_tasks.json`
- solo contiene la predicción;
- es el “resultado limpio” de la ejecución.

## `predicted_database.json`
- contiene base + predicción;
- es útil cuando quieres tratar la predicción como si ya formara parte de la base.

---

## 10. Archivos donde aparece toda esta información

## Frontend
### 1. `src/routes/_layout/site.tsx`
Contiene:
- activación del **Real Mode**;
- estado de paneles laterales;
- montaje de `PredictionPanel`.

### 2. `src/components/Blueprint3D/PredictionPanel.tsx`
Contiene:
- la UI del panel de predicción;
- el `NumberInput` de semanas;
- el `fetch` real a `/api/v1/prediction/predict_week`;
- el mensaje de éxito/error mostrado al usuario.

### 3. `src/services/predictionService.ts`
Contiene:
- una abstracción manual del endpoint `predict_week`.

Observación:
- este servicio existe, pero en la práctica el panel usa un `fetch` directo y no este wrapper.

### 4. `src/main.tsx`
Contiene:
- configuración de `OpenAPI.BASE`;
- lectura del token desde `localStorage`.

### 5. `vite.config.ts`
Contiene:
- el proxy de `/api` y `/static` hacia el backend.

### 6. `.env`
Contiene:
- `VITE_API_URL`, usada para redirigir las llamadas al backend.

---

## Backend
### 7. `app/main.py`
Contiene:
- la creación de la app FastAPI;
- el prefijo global `/api/v1`;
- el montaje de `/static`.

### 8. `app/api/main.py`
Contiene:
- el registro del router de predicción con prefijo `/prediction`.

### 9. `app/api/routes/prediction.py`
Contiene:
- las rutas HTTP reales de predicción;
- lectura y escritura de JSON;
- carga de modelos cacheados;
- bucle de predicción multi-semana;
- guardado de `predicted_tasks.json` y `predicted_database.json`.

### 10. `app/api/deps.py`
Contiene:
- la validación del JWT;
- la dependencia `CurrentUser` que protege la ruta de predicción.

### 11. `app/api/routes/login.py`
Contiene:
- la ruta `/login/access-token` que entrega el JWT necesario para llamar a predicción.

### 12. `app/api/prediction/config.py`
Contiene:
- configuración del motor de predicción.

### 13. `app/api/prediction/data/io.py`
Contiene:
- lectura de JSON histórico y conversión a DataFrame.

### 14. `app/api/prediction/data/preprocessing.py`
Contiene:
- preparación de semanas;
- features globales;
- contexto temporal por ocurrencia.

### 15. `app/api/prediction/models/occurrence_model.py`
Contiene:
- el modelo de ocurrencias.

### 16. `app/api/prediction/models/temporal_model.py`
Contiene:
- el modelo temporal.

### 17. `app/api/prediction/models/blocks.py`
Contiene:
- el encoder secuencial GRU bidireccional con atención usado por ambos modelos.

### 18. `app/api/prediction/predict.py`
Contiene:
- la inferencia real semana a semana;
- generación de candidatos temporales;
- reparación de solapes;
- materialización a timestamps.

### 19. `app/api/prediction/train.py`
Contiene:
- el pipeline de entrenamiento;
- guardado de checkpoints;
- resúmenes de métricas por semana.

### 20. `app/api/prediction/checkpoints/occurrence_model.pt`
Checkpoint del modelo de ocurrencias.

### 21. `app/api/prediction/checkpoints/temporal_model.pt`
Checkpoint del modelo temporal.

### 22. `app/api/prediction/aux_database_1.json`
Base histórica usada por el endpoint.

### 23. `app/api/prediction/predicted_tasks.json`
Salida solo de predicción.

### 24. `app/api/prediction/predicted_database.json`
Salida combinada histórico + predicción.

---

## 11. Observaciones importantes detectadas en la implementación actual

## 11.1. El frontend tiene duplicación
Existen dos caminos para llamar a la predicción:

- `PredictionPanel.tsx` hace `fetch` manual.
- `src/services/predictionService.ts` también implementa el endpoint.

Ahora mismo eso duplica lógica.

---

## 11.2. El cliente OpenAPI del frontend no parece estar actualizado con predicción
En la revisión hecha, la parte autogenerada del cliente (`src/client/...`) no muestra integración real del módulo de predicción.

Consecuencia:
- la feature se consume manualmente;
- hay riesgo de desalineación de tipos.

---

## 11.3. Hay desajuste entre tipos del frontend y respuesta real del backend
En el frontend, la interfaz `PredictWeekResponse` espera que cada elemento de `data` tenga:

- `task_name: string`
- `type: string`
- `week_offset: number`

Pero el backend devuelve realmente objetos con campos como:

- `uid`
- `device_uid`
- `task_name` (a menudo `null`)
- `type`
- `status`
- `start_time`
- `end_time`
- `mileage`
- `misc`
- `waypoints`

Y además **no devuelve `week_offset`** dentro de cada tarea.

Eso es importante porque el tipo del frontend no representa exactamente la respuesta real.

---

## 11.4. La salida se guarda en disco, pero no se consume visualmente aquí
El panel informa del éxito y del archivo guardado, pero en el código revisado no aparece una integración directa que:

- lea `predicted_tasks.json`;
- pinte las tareas predichas en la interfaz 3D;
- o refresque automáticamente una vista con la predicción.

---

## 11.5. Los modelos se cachean en memoria
`get_models()` usa `_models_cache`, por lo que:

- la primera petición carga modelos desde disco;
- las siguientes reutilizan esos mismos modelos ya cargados.

Ventaja:
- mejora el rendimiento.

Implicación:
- si sustituyes los checkpoints en disco mientras el servidor sigue vivo, puede que necesites reiniciar backend para forzar recarga limpia.

---

## 11.6. El endpoint predice siempre desde `aux_database_1.json`
La entrada principal del endpoint no es dinámica. Arranca siempre desde:

```text
app/api/prediction/aux_database_1.json
```

Eso significa que, tal como está, el usuario no selecciona desde frontend qué base usar.

---

## 11.7. El resultado combinado no reemplaza la base original
La base original usada para arrancar sigue siendo `aux_database_1.json`.

La predicción combinada se guarda aparte en:

```text
predicted_database.json
```

Por tanto, una nueva llamada volverá a partir de la base original, no de la combinada anterior, salvo que el código se cambie explícitamente.

---

## 12. Resumen ejecutivo

La funcionalidad de predicción del proyecto hace esto:

1. El usuario entra en **Real Mode**.
2. Abre el panel **Predicción**.
3. Indica cuántas semanas quiere generar.
4. El frontend manda una petición autenticada al backend.
5. El backend lee `aux_database_1.json`.
6. Usa dos modelos PyTorch ya entrenados:
   - uno para **cuántas tareas** habrá;
   - otro para **cuándo ocurren y cuánto duran**.
7. Genera la semana o semanas futuras de forma recursiva.
8. Guarda:
   - `predicted_tasks.json`
   - `predicted_database.json`
9. Devuelve un JSON al frontend con el resumen y las tareas generadas.

---

## 13. Archivos clave mínimos que debes conocer

Si solo quieres ubicar rápido toda la lógica de predicción, revisa estos primero:

### Front
- `src/routes/_layout/site.tsx`
- `src/components/Blueprint3D/PredictionPanel.tsx`
- `src/main.tsx`
- `vite.config.ts`

### Back
- `app/api/routes/prediction.py`
- `app/api/prediction/predict.py`
- `app/api/prediction/config.py`
- `app/api/prediction/data/io.py`
- `app/api/prediction/data/preprocessing.py`
- `app/api/prediction/models/occurrence_model.py`
- `app/api/prediction/models/temporal_model.py`
- `app/api/prediction/checkpoints/occurrence_model.pt`
- `app/api/prediction/checkpoints/temporal_model.pt`
- `app/api/prediction/aux_database_1.json`
- `app/api/prediction/predicted_tasks.json`
- `app/api/prediction/predicted_database.json`
