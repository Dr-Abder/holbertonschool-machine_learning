#!/usr/bin/env python3
"""
Ce module implémente l’opération de pooling (sous-échantillonnage)
sur un lot d’images multi-canaux, en supportant le max pooling
et l’average pooling.
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Applique une opération de pooling (max ou average) sur un lot d’images.

    Args:
    images (numpy.ndarray): Tableau de forme (m, h, w, c) contenant m images,
                                de hauteur h, largeur w et c canaux.
        kernel_shape (tuple): Dimensions (kh, kw) de la fenêtre de pooling.
        stride (tuple): Pas de déplacement (sh, sw) de la fenêtre.
        mode (str): Type de pooling :
            - 'max' → max pooling (valeur maximale),
            - 'avg' → average pooling (moyenne des valeurs).

    Returns:
        numpy.ndarray: Tableau de forme (m, h_out, w_out, c), où :
            - h_out = (h - kh) // sh + 1
            - w_out = (w - kw) // sw + 1
        Chaque canal est réduit en fonction du mode choisi.
    """
    # 1. Récupérer les dimensions
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # 2. Calculer les dimensions de sortie (pas de padding en pooling)
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c))

    # 3. Boucles (2 seulement : i et j)
    for i in range(output_h):
        for j in range(output_w):
            # Extraire la région
            region = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]

            # Appliquer max ou avg selon le mode
            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            else:  # mode == 'avg'
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
