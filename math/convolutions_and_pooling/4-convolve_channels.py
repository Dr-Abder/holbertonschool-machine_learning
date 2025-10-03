#!/usr/bin/env python3
"""
Ce module implémente une convolution sur des images multi-canaux
(par ex. RGB), avec gestion du padding et du stride.
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Effectue une convolution sur un lot d’images multi-canaux
    (par exemple RGB) avec un noyau donné, en tenant compte
    du padding et du stride.

    Args:
        images (numpy.ndarray): Tableau de forme (m, h, w, c) contenant m images
                                avec h hauteur, w largeur et c canaux.
        kernel (numpy.ndarray): Noyau de convolution de forme (kh, kw, kc),
                                kc devant être égal à c.
        padding (str ou tuple):
            - 'same'  → padding automatique pour conserver la taille.
            - 'valid' → pas de padding.
            - (ph, pw) → padding manuel en hauteur et largeur.
        stride (tuple): (sh, sw) indiquant le pas de la convolution en
                        hauteur et largeur.

    Returns:
        numpy.ndarray: Tableau de forme (m, h_out, w_out) contenant les
                       résultats de la convolution.
                       - h_out = ((h + 2*ph - kh) // sh) + 1
                       - w_out = ((w + 2*pw - kw) // sw) + 1
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_images = np.zeros((m, h + ph * 2, w + pw * 2, c))
    padded_images[:, ph:ph + h, pw:pw + w, :] = images

    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = padded_images[:, i * sh:i * sh + kh,
                                   j * sw:j * sw + kw, :]
            output[:, i, j] = (region * kernel).sum(axis=(1, 2, 3))

    return output
