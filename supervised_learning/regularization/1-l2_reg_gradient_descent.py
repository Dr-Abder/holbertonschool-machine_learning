#!/usr/bin/env python3

import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    
    m = Y.shape[1]
    for i in range (L, 0, -1):
        if i == L:
            dZ = cache[f'A{L}'] - Y
        
        else:
            dZ = (weights[f'W{i+1}'].T @ dZ) * (1 - cache[f'A{i}']**2)

        dW = dZ  @ cache[f'A{i-1}'].T
        db = np.sum(dZ, axis=1, keepdims=True)
        dW += (lambtha / m) * weights[f'W{i}']
        weights[f'W{i}'] = weights[f'W{i}'] - alpha * dW # formule de mise à jour
        weights[f'b{i}'] = weights[f'b{i}'] - alpha * db # formule de mise à jour