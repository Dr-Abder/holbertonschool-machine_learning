#!/usr/bin/env python3

"""
Effectue la rétropropagation à travers une couche de pooling.

Ce module calcule les gradients des entrées d'une couche de pooling
(max ou moyenne) en fonction du gradient d'erreur reçu de la couche
suivante. Il est utilisé dans le processus de rétropropagation pour
l'entraînement des réseaux de neurones convolutionnels.
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Calcule la rétropropagation du gradient à travers une couche de pooling.

    Parameters
    ----------
    dA : np.ndarray
        Gradient du coût par rapport à la sortie du pooling
        de forme (m, h_new, w_new, c)
    A_prev : np.ndarray
        Entrée de la couche précédente
        de forme (m, h_prev, w_prev, c)
    kernel_shape : tuple
        Taille du noyau de pooling (kh, kw)
    stride : tuple, optional
        Pas de déplacement (sh, sw), par défaut (1, 1)
    mode : str, optional
        Type de pooling à appliquer ('max' ou 'avg'), par défaut 'max'

    Returns
    -------
    np.ndarray
        Gradient du coût par rapport à l'entrée de la couche précédente
        (m, h_prev, w_prev, c)
    """
    m, h_prev, w_prev, c = A_prev.shape
    _, h_new, w_new, _ = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c):
                region = A_prev[
                    :,
                    i * sh:i * sh + kh,
                    j * sw:j * sw + kw,
                    k
                ]

                if mode == 'max':
                    max_value = np.max(region, axis=(1, 2), keepdims=True)
                    mask = (region == max_value).astype(float)
                    dA_prev[
                        :,
                        i * sh:i * sh + kh,
                        j * sw:j * sw + kw,
                        k
                    ] += mask * dA[:, i, j, k][:, np.newaxis, np.newaxis]

                elif mode == 'avg':
                    avg_value = dA[:, i, j, k] / (kh * kw)
                    avg_gradient = avg_value[:, np.newaxis, np.newaxis]
                    dA_prev[
                        :,
                        i * sh:i * sh + kh,
                        j * sw:j * sw + kw,
                        k
                    ] += avg_gradient

    return dA_prev
