#!/usr/bin/env python3
"""
Ce module implémente une convolution sur un lot d’images multi-canaux
avec plusieurs noyaux (comme dans une couche de réseau CNN).
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Applique une convolution sur un lot d’images multi-canaux avec
    plusieurs noyaux de convolution (filtres).

    Args:
    images (numpy.ndarray): Tableau de forme (m, h, w, c) contenant m images,
                                avec hauteur h, largeur w et c canaux.
    kernels (numpy.ndarray): Noyaux de convolution de forme (kh, kw, kc, nc),
                            avec :
                            - kh, kw = dimensions du noyau,
                            - kc = nombre de canaux (doit correspondre à c),
                            - nc = nombre de noyaux (filtres).
        padding (str ou tuple):
            - 'same'  → padding automatique pour conserver la taille,
            - 'valid' → pas de padding,
            - (ph, pw) → padding manuel (haut/bas, gauche/droite).
        stride (tuple): (sh, sw) indiquant le pas de convolution
                        en hauteur et largeur.

    Returns:
        numpy.ndarray: Tableau de forme (m, h_out, w_out, nc), où chaque
                       canal de sortie correspond à la convolution avec
                       un noyau différent.
                       - h_out = ((h + 2*ph - kh) // sh) + 1
                       - w_out = ((w + 2*pw - kw) // sw) + 1
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
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
    output = np.zeros((m, output_h, output_w, nc))

    for k in range(nc):
        for i in range(output_h):
            for j in range(output_w):
                current_kernel = kernels[:, :, :, k]
                region = padded_images[:, i * sh:i * sh + kh,
                                       j * sw:j * sw + kw, :]
                output[:, i, j, k] = (region * current_kernel).sum(axis=(1,
                                                                   2, 3))

    return output
