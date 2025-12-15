#!/usr/bin/env python3
"""
Optimizing k - Kmeans
"""

import numpy as np

def optimum_k(X, kmin=1, kmax=None, iterations=1000):

    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = kmin+1
    else:
        if not isinstance(kmax, int) or kmax <= 0:
            return None, None
    
    if kmin >= kmax:
        return None, None

    results = []
    d_vars = []

    C_ref, clss_ref = kmeans(X, kmin, iterations)
    var_ref = variance(X, C_ref)

    for k in range(kmin, kmax+1):

        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        var_k = variance(X, C)
        delta = var_ref - var_k
        d_vars.append(delta)
    return results, d_vars