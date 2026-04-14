# Análisis de datasets

## Resumen global

- eventos totales: **161443**
- bases de datos: **8**
- robots/dispositivos: **24**
- tareas: **4**
- rango temporal agregado: **2026-01-05 09:00:00+01:00 → 2036-01-05 00:25:00+01:00**

## Lectura técnica del conjunto completo

### 1) Hay dos familias de bases

**Bases rígidas / compactas**
- `aux_database`
- `aux_database_1`
- `aux_database_2`
- `robot_schedule_v4_2025_2026`

Comparten menos semanas, menor volumen y horarios relativamente repetitivos. Aquí conviene regularizar para **evitar memorizar semanas concretas** y para **anclar más fuerte a la plantilla**.

**Bases con mayor volatilidad temporal o de carga**
- `nexus_schedule_10years`
- `nexus_schedule_5years`
- `robot_schedule_5years`

Aquí conviene suavizar objetivos y permitir residual más flexible, porque la penalización dura al bin exacto o al delta exacto sobrecastiga variaciones que son normales del dataset.

### 2) `robot_schedule_gamma` es un caso especial

Es la base con más masa histórica y más robots. No necesita la regularización más fuerte por escasez, pero sí control para no **sobreajustar combinaciones robot-base** ni producir distribuciones excesivamente confiadas.

### 3) Las duraciones están muy discretizadas

En la práctica, las duraciones son casi deterministas por tarea. Eso sugiere que la regularización útil debe concentrarse en:

- **conteos por tarea**,
- **día y hora de inicio**,
- **desviación frente a plantilla**,
- **suavidad de las distribuciones de tiempo**.

## Perfil resumido por base

### aux_database
- 8975 eventos, 104 semanas, 2 robots
- CV semanal: **0.088**
- Perfil: tamaño medio-corto, estabilidad razonable
- Regularización útil: ruido de entrada moderado, smoothing moderado y anclaje temporal medio

### aux_database_1
- 4712 eventos, 104 semanas, 1 robot
- CV semanal: **0.084**
- Perfil: menos masa que `aux_database`, algo más sensible a sobreajuste
- Regularización útil: más dropout/ruido y más smoothing que `aux_database`

### aux_database_2
- 4263 eventos, 104 semanas, 1 robot
- CV semanal: **0.054**
- Perfil: base pequeña pero especialmente estable
- Regularización útil: **anclaje fuerte a plantilla** y shrinkage más alto del residual

### nexus_schedule_10years
- 32732 eventos, 522 semanas, 1 robot
- CV semanal: **0.266**
- Perfil: mucha historia, pero mayor volatilidad de carga
- Regularización útil: soft targets de delta y de hora, menos castigo al desplazamiento pequeño, menor anchor penalty

### nexus_schedule_5years
- 16368 eventos, 261 semanas, 1 robot
- CV semanal: **0.267**
- Perfil: misma familia que `nexus_schedule_10years`, con menos historia
- Regularización útil: smoothing más fuerte que en `nexus_schedule_10years`, pero manteniendo flexibilidad residual

### robot_schedule_5years
- 19542 eventos, 261 semanas, 2 robots
- CV semanal: **0.218**
- Perfil: intermedio entre rigidez y volatilidad
- Regularización útil: mezcla equilibrada de priors temporales, soft targets y shrinkage medio

### robot_schedule_gamma
- 72495 eventos, 111 semanas, 15 robots
- CV semanal: **0.114**
- Perfil: gran volumen y diversidad de robots
- Regularización útil: control de sobreconfianza, dropout moderado y prior temporal relativamente fuerte

### robot_schedule_v4_2025_2026
- 2356 eventos, 52 semanas, 1 robot
- CV semanal: **0.087**
- Perfil: base más corta y frágil
- Regularización útil: **la más fuerte** de todo el proyecto en ruido de entrada, smoothing y prior temporal

## Traducción de estos hallazgos a la implementación

La implementación nueva convierte esta lectura en perfiles automáticos de regularización:

- **escasez** → más dropout, más ruido, más label smoothing
- **volatilidad** → soft targets más anchos para delta y hora
- **rigidez** → más peso al prior y mayor anchor regularization

Esto evita usar la misma pérdida para bases con comportamientos claramente distintos.
