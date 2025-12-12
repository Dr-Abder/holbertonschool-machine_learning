#!/usr/bin/env python3
import numpy as np


def kmeans(X, k, iterations=1000):

    if not isinstance(X, np.ndarray):
        return None
    if not isinstance(k , int) or k <= 0:
        return None
    if len(X.shape) != 2:
        return None

    n, d = X.shape

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    C = np.random.uniform(mins, maxs, size=(k, d))

    for i in range(iterations):
        old_C = C.copy()

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            points_in_cluster = X[clss == j]

            if len(points_in_cluster) == 0:
                C[j] = np.random.uniform(mins, maxs, size=d)
            
            else:
                C[j] = np.mean(points_in_cluster, axis=0)

        if np.array_equal(old_C, C):
            break

        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

    return C, clss