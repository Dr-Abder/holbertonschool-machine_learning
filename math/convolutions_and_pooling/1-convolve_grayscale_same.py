#!/usr/bin/env python3
"""
Ce module implémente une fonction de convolution sur des images en niveaux
de gris en utilisant le mode "same". Dans ce mode, on ajoute du padding
autour de l’image pour que la taille de la sortie soit identique à celle
de l’entrée.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Effectue une convolution "same" sur un lot d’images en niveaux de gris.

    Dans ce mode, un padding est ajouté autour de l’image afin que la
    taille de sortie reste égale à la taille d’entrée.

    Args:
        images (numpy.ndarray): Tableau de forme (m, h, w) contenant m images,
                                chacune de taille (h, w).
        kernel (numpy.ndarray): Noyau de convolution de forme (kh, kw).

    Returns:
        numpy.ndarray: Tableau de forme (m, h, w) contenant les images
                       transformées après convolution.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calcul du padding nécessaire (haut/bas et gauche/droite)
    pad_h = kh // 2
    pad_w = kw // 2

    # Ajout du padding autour de chaque image
    padded_images = np.zeros((m, h + 2 * pad_h, w + 2 * pad_w))
    padded_images[:, pad_h:pad_h + h, pad_w:pad_w + w] = images

    output_padded = np.zeros((m, h, w))

    # Convolution
    for i in range(h):
        for j in range(w):
            region = padded_images[:, i:i + kh, j:j + kw]
            output_padded[:, i, j] = (region * kernel).sum(axis=(1, 2))

    return output_padded
