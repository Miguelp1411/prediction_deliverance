import numpy as np
from scipy.optimize import linear_sum_assignment

BIN_MINUTES = 5


def task_cost(real, pred):

    cost = 0

    if real["task_name"] != pred["task_name"]:
        cost += 100

    cost += abs(real["start_bin"] - pred["start_bin"]) * BIN_MINUTES
    cost += abs(real["duration"] - pred["duration"])

    return cost


def hungarian_match(true_tasks, pred_tasks):

    n = len(true_tasks)
    m = len(pred_tasks)

    size = max(n, m)

    cost_matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):

            if i < n and j < m:
                cost_matrix[i, j] = task_cost(true_tasks[i], pred_tasks[j])
            else:
                cost_matrix[i, j] = 200  # penalización dummy

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    pairs = []

    for r, c in zip(row_ind, col_ind):

        if r < n and c < m:
            pairs.append((true_tasks[r], pred_tasks[c]))

    return pairs