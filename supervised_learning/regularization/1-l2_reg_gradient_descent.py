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

          math:

              dW = (1/m) * (dZ @ A^{[l-1]}^T) + (λ/m) * W^{[l]}
    """

    m = Y.shape[1]
    A_prev = cache['A' + str(L - 1)]
    A_L = cache['A' + str(L)]

    dZ = A_L - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # update après calcul
        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db

        if layer > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - A_prev ** 2)
