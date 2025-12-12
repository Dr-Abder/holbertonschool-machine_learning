#!/usr/bin/env python3

"""
Initialize cluster centroids
"""

import numpy as np


def initialize(X, k):

    if not isinstance(X, np.ndarray):
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    if len(X.shape) != 2:
        return None

    n, d = X.shape

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    centroids = np.random.uniform(mins, maxs, size=(k, d))

    return centroids