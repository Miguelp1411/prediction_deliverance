# Training Report: OccurrenceResidual

- **Total epochs**: 2
- **Best epoch**: 2
- **Training time**: 3.3s

## Best Metrics

| Metric | Value |
|--------|-------|
| best_val_loss | 1.7192 |

## Baseline Comparison

| Baseline | Metric | Value |
|----------|--------|-------|
| retrieval_topk | task_precision | 100.0000 |
| retrieval_topk | task_recall | 99.4318 |
| retrieval_topk | task_f1 | 99.7126 |
| retrieval_topk | time_exact_accuracy | 96.5116 |
| retrieval_topk | time_close_accuracy_5m | 96.5116 |
| retrieval_topk | time_close_accuracy_10m | 96.5116 |
| retrieval_topk | start_mae_minutes | 3.8372 |
| retrieval_topk | duration_mae_minutes | 0.0000 |
| retrieval_topk | overlap_same_device_count | 66.2500 |
| retrieval_topk | overlap_global_count | 66.2500 |
| retrieval_topk | matched_pairs | 45.0000 |
| retrieval_topk | total_true | 45.2500 |
| retrieval_topk | total_pred | 45.0000 |

## Learning Curve

| Epoch | Train Loss | Val Loss | Monitor |
|-------|------------|----------|---------|
| 1 | 1.9300 | 1.8300 | 1.8300 |
| 2 | 1.8205 | 1.7192 | 1.7192 |
