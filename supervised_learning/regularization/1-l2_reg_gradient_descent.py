#!/usr/bin/env python3
"""
Module qui réalise la descente de gradient avec régularisation L2.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Met à jour les poids d’un réseau de neurones avec régularisation L2
    en utilisant la descente de gradient.

    Args:
        Y (numpy.ndarray): Tableau des étiquettes au format one-hot,
            de forme (classes, m).
        weights (dict): Dictionnaire contenant les poids et biais du réseau.
            Les clés sont de la forme 'Wl' et 'bl' pour la couche l.
        cache (dict): Dictionnaire contenant les sorties de chaque couche
            (les activations 'Al').
        alpha (float): Taux d’apprentissage.
        lambtha (float): Paramètre de régularisation (lambda).
        L (int): Nombre de couches du réseau.

    Returns:
        None: La fonction met à jour le dictionnaire `weights` en place.

    Notes:
        - Utilise le cache direct (activations) pour calculer les gradients
          en sens inverse.
        - Applique la dérivée de la fonction d’activation tanh
          pour les couches cachées.
        - Ajoute le terme de régularisation L2 au gradient :

          .. math::

              dW = (1/m) * (dZ @ A^{[l-1]}^T) + (λ/m) * W^{[l]}
    """
    m = Y.shape[1]
    for i in range(L, 0, -1):
        if i == L:
            dZ = cache[f'A{L}'] - Y
        else:
            dZ = (weights[f'W{i+1}'].T @ dZ) * (1 - cache[f'A{i}']**2)

        dW = dZ @ cache[f'A{i-1}'].T / m
        db = np.mean(dZ, axis=1, keepdims=True)
        dW += (lambtha / m) * weights[f'W{i}']
        weights[f'W{i}'] = weights[f'W{i}'] - alpha * dW
        weights[f'b{i}'] = weights[f'b{i}'] - alpha * db
