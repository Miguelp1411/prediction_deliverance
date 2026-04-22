# Mejoras aplicadas en v5

Se ha actualizado el proyecto para reducir sobreajuste en occurrence y temporal mediante:

- regularización explícita en occurrence (`delta_reg_weight`)
- regularización equivalente en temporal (penalización de confianza y desviación respecto al anchor)
- label smoothing en ambos modelos
- gradient clipping
- scheduler de learning rate por plateau
- encoder GRU corregido para usar hidden states finales bidireccionales
- dropout adicional tras el encoder
- reducción de capacidad por defecto en ambos modelos
- normalización train-only de history, numeric features y candidate features
- persistencia de la normalización en `config.feature_scaling` dentro de los checkpoints
- eliminación de señales temporales absolutas en occurrence
- ablation configurable de `lag_52` (desactivada por defecto)
- reducción del espacio candidato temporal por defecto para mejorar robustez
