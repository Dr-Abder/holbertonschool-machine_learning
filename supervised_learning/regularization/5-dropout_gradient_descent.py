#!/usr/bin/env python3
"""
Ce module implémente la rétropropagation (gradient descent)
d’un réseau de neurones avec régularisation par Dropout.
Le Dropout permet de réduire le surapprentissage en forçant
le réseau à ne pas dépendre uniquement de certains neurones.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Met à jour les poids et biais d’un réseau de neurones
    en utilisant la descente de gradient avec Dropout.

    Args:
        Y (numpy.ndarray): Labels attendus de forme (classes, m),
                           où m est le nombre d’exemples.
        weights (dict): Dictionnaire contenant les poids et biais
                        du réseau. Clés : 'Wl', 'bl' pour chaque couche l.
        cache (dict): Dictionnaire contenant les activations ('Al')
                      et les masques Dropout ('Dl') de chaque couche
                      obtenus pendant la propagation avant.
        alpha (float): Taux d’apprentissage.
        keep_prob (float): Probabilité de conserver un neurone actif
                           pendant le Dropout (0 < keep_prob ≤ 1).
        L (int): Nombre total de couches du réseau.

    Returns:
        None: Les poids et biais sont mis à jour directement
              dans le dictionnaire `weights`.
    """
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y
    weights = {}

    weights = {}
    for i in range(1, L+1):
        n_prev = layer_dims[i-1]
        n_l = layer_dims[i]
        weights[f'W{i}'] = np.random.randn(n_l, n_prev) * np.sqrt(2 / n_prev)
        weights[f'b{i}'] = np.zeros((n_l, 1))

        if i > 1:
            dA_prev = weights[f'W{i}'].T @ dZ
            dA_prev = dA_prev * cache[f'D{i-1}'] / keep_prob
            dZ = dA_prev * (1 - cache[f'A{i-1}']**2)
