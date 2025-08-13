#!/usr/bin/env python3
"""
Fonction qui concatène deux matrice
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Parameters:
    - mat1: matrice
    - mat2: matrice
    - axis: l'axe pour voir dans qu'elle sens sera la concaténation

    Returns:
    - Une nouvelle matrice de mat1 + mat2 qui à été concaténer selon
    l'axe , soit concaténation verticale du coup empilement de lignes si axis = 0
    sinon concaténation horizontale du coup extension des colonnes si axis = 1
    """
    return np.concatenate((mat1, mat2), axis=axis)
