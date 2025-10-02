#!/usr/bin/env python3
"""
Ce module implémente une convolution sur des images en niveaux de gris
avec un padding défini manuellement (padding arbitraire).
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Effectue une convolution sur un lot d’images en niveaux de gris
    avec un padding spécifié par l’utilisateur.

    Args:
        images (numpy.ndarray): Tableau de forme (m, h, w) contenant m images,
                                chacune de taille (h, w).
        kernel (numpy.ndarray): Noyau de convolution de forme (kh, kw).
        padding (tuple): (ph, pw) indiquant le nombre de pixels de padding
                         à ajouter en hauteur et en largeur.

    Returns:
        numpy.ndarray: Tableau de forme (m, h_out, w_out) contenant
                       les images transformées après convolution.
                       - h_out = h + 2*ph - kh + 1
                       - w_out = w + 2*pw - kw + 1
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Ajout du padding autour de l’image
    padded_images = np.zeros((m, h + 2 * ph, w + 2 * pw))
    padded_images[:, ph:ph + h, pw:pw + w] = images

    # Dimensions de sortie
    output_h = (h + 2 * ph) - kh + 1
    output_w = (w + 2 * pw) - kw + 1
    output = np.zeros((m, output_h, output_w))

    # Convolution
    for i in range(output_h):
        for j in range(output_w):
            region = padded_images[:, i:i + kh, j:j + kw]
            output[:, i, j] = (region * kernel).sum(axis=(1, 2))

    return output
