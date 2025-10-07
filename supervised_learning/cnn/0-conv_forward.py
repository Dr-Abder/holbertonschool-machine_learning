#!/usr/bin/env python3
"""
Effectue la propagation avant d’une couche de convolution.

Cette fonction applique une opération de convolution entre les entrées `A_prev` 
et un ensemble de filtres `W`, ajoute le biais correspondant `b`, puis applique 
une fonction d’activation donnée sur le résultat.  
Elle prend en charge les modes de padding `"same"` et `"valid"`, ainsi que 
le paramètre de stride.

Ce type d’opération correspond à la propagation avant d’une couche de 
réseau de neurones convolutionnel (CNN).
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Effectue la propagation avant à travers une couche de convolution.

    Args:
        A_prev (numpy.ndarray): tenseur d’entrée ou activations de la couche précédente, 
            de forme (m, h_prev, w_prev, c_prev)
            - m : nombre d’exemples
            - h_prev, w_prev : hauteur et largeur de l’entrée
            - c_prev : nombre de canaux de l’entrée
        W (numpy.ndarray): tenseur des poids (filtres) de forme (kh, kw, c_prev, c_new)
            - kh, kw : hauteur et largeur de chaque filtre
            - c_prev : nombre de canaux d’entrée
            - c_new : nombre de filtres (canaux de sortie)
        b (numpy.ndarray): biais, de forme (1, 1, 1, c_new)
        activation (callable): fonction d’activation à appliquer (ex: relu, sigmoid, tanh)
        padding (str): type de padding à utiliser
            - "same" : conserve les dimensions spatiales d’entrée
            - "valid" : aucune bordure ajoutée, la sortie rétrécit
        stride (tuple): tuple (sh, sw) représentant les pas de déplacement en hauteur et largeur

    Returns:
        numpy.ndarray: tenseur de sortie contenant les activations, de forme (m, h_new, w_new, c_new)
            avec :
                h_new = ((h_prev + 2*ph - kh) // sh) + 1  
                w_new = ((w_prev + 2*pw - kw) // sw) + 1

    Raises:
        ValueError: si le paramètre `padding` n’est ni "same" ni "valid".

    Notes:
        - L’implémentation utilise des boucles explicites pour plus de clarté pédagogique.
        - Elle n’est pas optimisée (pas de vectorisation avancée).
        - Le padding est calculé de façon à conserver les dimensions d’entrée 
          lorsque `padding="same"`.
        - Cette fonction illustre le fonctionnement fondamental d’une couche de convolution 
          avant toute optimisation GPU.

    Exemple:
        >>> import numpy as np
        >>> def relu(x): return np.maximum(0, x)
        >>> A_prev = np.random.randn(2, 5, 5, 3)
        >>> W = np.random.randn(3, 3, 3, 8)
        >>> b = np.zeros((1, 1, 1, 8))
        >>> A = conv_forward(A_prev, W, b, relu, padding="same", stride=(1, 1))
        >>> A.shape
        (2, 5, 5, 8)
    """
    m, h, w, c = A_prev.shape
    kh, kw, _, kc = W.shape
    sh, sw = stride

    # Étape 1 : Calcul du padding
    if padding == "valid":
        ph = 0
        pw = 0
    elif padding == "same":
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    else:
        raise ValueError("padding doit être 'same' ou 'valid'")

    # Étape 2 : Appliquer le padding
    A_prev_padded = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Étape 3 : Calcul des dimensions de sortie
    _, padded_h, padded_w, _ = A_prev_padded.shape
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1

    # Étape 4 : Initialiser la sortie
    A = np.zeros((m, output_h, output_w, kc))

    # Étape 5 : Effectuer la convolution
    for img in range(m):
        for i in range(output_h):
            for j in range(output_w):
                for k in range(kc):
                    kernel = W[:, :, :, k]
                    region = A_prev_padded[
                        img,
                        i * sh : i * sh + kh,
                        j * sw : j * sw + kw,
                        :
                    ]
                    conv_value = (region * kernel).sum() + b[0, 0, 0, k]
                    A[img, i, j, k] = activation(conv_value)

    return A
