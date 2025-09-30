#!/usr/bin/env python3
"""
Ce module implémente une fonction de convolution sur des images en niveaux
de gris en utilisant le mode "valid". Dans ce mode, le noyau se déplace
uniquement là où il peut s’appliquer entièrement sans dépassement,
réduisant ainsi la taille de l’image de sortie.
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Effectue une convolution "valid" sur un lot d’images en niveaux de gris.

    Args:
        images (numpy.ndarray): Tableau de forme (m, h, w) contenant m images,
                                chacune de taille (h, w).
        kernel (numpy.ndarray): Noyau de convolution de forme (kh, kw).

    Returns:
    numpy.ndarray: Tableau de forme (m, h - kh + 1, w - kw + 1) contenant
                les images transformées après application de la convolution.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1
    output_size = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = images[:, i:i+kh, j:j+kw]
            output_size[:, i, j] = (region * kernel).sum(axis=(1, 2))

    return output_size
