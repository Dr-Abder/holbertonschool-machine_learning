#!/usr/bin/env python3
"""
Ce module implémente la propagation avant (forward propagation)
d’un réseau de neurones avec la régularisation par Dropout.
Le Dropout permet de réduire le surapprentissage en désactivant
aléatoirement une proportion de neurones lors de l’entraînement.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Réalise la propagation avant d’un réseau de neurones avec Dropout.

    Args:
        X (numpy.ndarray): Données d’entrée de forme (nx, m),
                           où nx est le nombre de caractéristiques
                           et m le nombre d’exemples.
        weights (dict): Dictionnaire contenant les poids et biais
                        des différentes couches du réseau.
                        Clés : 'Wl', 'bl' pour chaque couche l.
        L (int): Nombre total de couches du réseau.
        keep_prob (float): Probabilité de conserver un neurone actif
                           pendant le Dropout (0 < keep_prob ≤ 1).

    Returns:
        dict: Un dictionnaire `cache` contenant :
              - 'A0' : entrée initiale X
              - 'Al' : activation de chaque couche l
              - 'Dl' : masque binaire appliqué lors du Dropout
                       (pour les couches cachées uniquement).
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L+1):

        A_prev = cache[f'A{i-1}']
        Z = weights[f'W{i}'] @ A_prev + weights[f'b{i}']

        if i == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        else:
            A = np.tanh(Z)

        if i < L:
            D = np.random.binomial(1, keep_prob, A.shape)
            A = A * D / keep_prob
            cache[f'D{i}'] = D

        cache[f'A{i}'] = A

    return cache
