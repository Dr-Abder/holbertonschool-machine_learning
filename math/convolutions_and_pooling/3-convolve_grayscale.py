#!/usr/bin/env python3
"""
Ce module implémente une convolution sur des images en niveaux de gris,
en intégrant la gestion du padding ('same', 'valid' ou tuple) et du stride.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Effectue une convolution sur un lot d’images en niveaux de gris
    avec gestion flexible du padding et du stride.

    Args:
        images (numpy.ndarray): Tableau de forme (m, h, w) contenant m images
                                en niveaux de gris.
        kernel (numpy.ndarray): Noyau de convolution de forme (kh, kw).
        padding (str ou tuple): 
            - 'same'  → padding automatique pour conserver la même taille.
            - 'valid' → pas de padding.
            - (ph, pw) → padding manuel en hauteur et largeur.
        stride (tuple): (sh, sw) indiquant le pas de déplacement de la
                        fenêtre de convolution en hauteur et en largeur.

    Returns:
        numpy.ndarray: Tableau de forme (m, h_out, w_out) contenant les images
                       transformées après convolution.
                       - h_out = ((h + 2*ph - kh) // sh) + 1
                       - w_out = ((w + 2*pw - kw) // sw) + 1
    """
    # 1. Dimensions d’entrée et stride
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # 2. Calcul du padding
    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # 3. Ajout du padding
    padded_images = np.zeros((m, h + 2 * ph, w + 2 * pw))
    padded_images[:, ph:ph + h, pw:pw + w] = images

    # 4. Dimensions de sortie
    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1
    output = np.zeros((m, output_h, output_w))

    # 5. Convolution avec stride
    for i in range(output_h):
        for j in range(output_w):
            region = padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
            output[:, i, j] = (region * kernel).sum(axis=(1, 2))

    return output
