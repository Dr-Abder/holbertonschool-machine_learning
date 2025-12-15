#!/usr/bin/env python3


import numpy as np


def variance(X, C):

    if not isinstance(X , np.ndarray) or not isinstance(C, np.ndarray):
        return None

    if len(X.shape) != 2 or len(C.shape) != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    assigned_centroids = C[clss]

    diff = X - assigned_centroids
    squared_distances = np.sum(diff**2, axis=1)

    variance = np.sum(squared_distances)

    return variance