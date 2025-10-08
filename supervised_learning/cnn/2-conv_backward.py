#!/usr/bin/env python3
"""
Effectue la rétropropagation du gradient à travers une couche de convolution.

Cette fonction calcule les gradients des entrées, des poids et des biais
d'une couche de convolution, à partir du gradient d’erreur `dZ` reçu
de la couche suivante lors de la rétropropagation.
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Calcule la rétropropagation à travers une couche de convolution.

    Parameters
    ----------
    dZ : np.ndarray
        Gradient du coût par rapport à la sortie (m, h_new, w_new, c_new)
    A_prev : np.ndarray
        Entrée de la couche précédente (m, h_prev, w_prev, c_prev)
    W : np.ndarray
        Filtres de convolution (kh, kw, c_prev, c_new)
    b : np.ndarray
        Biais (1, 1, 1, c_new)
    padding : str, optional
        'same' ou 'valid' (par défaut 'same')
    stride : tuple, optional
        Pas de déplacement (par défaut (1, 1))

    Returns
    -------
    dA_prev : np.ndarray
        Gradient par rapport à l'entrée
    dW : np.ndarray
        Gradient par rapport aux poids
    db : np.ndarray
        Gradient par rapport aux biais
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, _ = W.shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    if padding == "valid":
        ph = pw = 0
    elif padding == "same":
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2

    A_prev_padded = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    dA_prev_padded = np.pad(
        dA_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                region = A_prev_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

                dZ_slice = dZ[:, i, j, k][:, np.newaxis,
                                          np.newaxis, np.newaxis]

                dW[:, :, :, k] += (region * dZ_slice).sum(axis=0)

                W_slice = W[:, :, :, k]
                dA_prev_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw,
                               :] += W_slice * dZ_slice

    if padding == "same":
        dA_prev = dA_prev_padded[:, ph:ph+h_prev, pw:pw+w_prev, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
