#!/usr/bin/env python3
"""
Implémente la propagation avant d’une couche de **pooling** 
dans un réseau de neurones convolutionnel (CNN).

Le pooling est une opération de réduction spatiale appliquée
après la convolution. Il permet de diminuer la taille des 
caractéristiques extraites tout en conservant les informations 
essentielles, et de réduire le risque de surapprentissage.

Cette fonction prend en charge deux types de pooling :
- **Max pooling** : conserve la valeur maximale d’une région.
- **Average pooling** : calcule la moyenne des valeurs d’une région.
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Effectue la propagation avant d’une couche de pooling (max ou moyenne).

    Args:
        A_prev (numpy.ndarray): tenseur d’entrée ou activations de la couche précédente,
            de forme (m, h_prev, w_prev, c)
            - m : nombre d’exemples (images)
            - h_prev, w_prev : hauteur et largeur de chaque image
            - c : nombre de canaux (features maps)
        kernel_shape (tuple): taille de la fenêtre de pooling (kh, kw)
        stride (tuple): pas de déplacement (sh, sw)
        mode (str): type de pooling à appliquer
            - "max" : sélectionne la valeur maximale dans chaque fenêtre
            - "avg" : calcule la moyenne des valeurs dans chaque fenêtre

    Returns:
        numpy.ndarray: tenseur de sortie après pooling, 
        de forme (m, h_new, w_new, c)
        où :
            h_new = (h_prev - kh) // sh + 1  
            w_new = (w_prev - kw) // sw + 1

    Notes:
        - Le pooling est appliqué indépendamment sur chaque canal.
        - Aucun padding n’est appliqué ici.
        - Le pooling réduit la dimension spatiale et rend les représentations
          plus robustes aux translations et distorsions locales.

    Exemple:
        >>> import numpy as np
        >>> A_prev = np.random.randn(2, 6, 6, 3)
        >>> A = pool_forward(A_prev, kernel_shape=(2, 2), stride=(2, 2), mode='max')
        >>> A.shape
        (2, 3, 3, 3)
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calcul des dimensions de sortie
    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1

    # Initialisation du tenseur de sortie
    A = np.zeros((m, output_h, output_w, c))

    # Application du pooling
    for i in range(output_h):
        for j in range(output_w):
            region = A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(region, axis=(1, 2))

    return A
