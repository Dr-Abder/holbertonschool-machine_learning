#!/usr/bin/env python3
"""
Calculae the normalization constants (mean and standard deviation)
of a matrix
"""

import numpy as np


def normalization_constants(X):
    """
    Calculae the normalization constants (mean and standard deviation)
    of a matrix
    """
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    return mean, std_dev
