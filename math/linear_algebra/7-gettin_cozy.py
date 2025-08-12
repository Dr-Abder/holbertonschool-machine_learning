#!/usr/bin/env python3
"""
Cette fonction concatène mat1 et mat2 qui sont des matrices
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    concatène mat1 et mat2
    """

    if axis == 0:
        if len(mat1[0]) == len(mat2[0]):
            return mat1 + mat2
        else:
            return None
        """
        Vérifie sur l'axe 0 (concaténation verticale, empilement de lignes)
        Si mat1 et mat2 ont le même nombre de colonnes,
        alors on les concatène en empilant mat2 sous mat1 (mat1 + mat2),
        sinon on retourne None.
        """

    if axis == 1:
        if len(mat1) == len(mat2):
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
        else:
            return None
        """
        Vérifie sur l'axe 1 (concaténation horizontale, extension des colonnes)
        Si mat1 et mat2 ont le même nombre de lignes,
        alors on concatène chaque ligne de mat1 avec la ligne correspondante de mat2 
        grace à la boucle qui passe de ligne en ligne pour chaque matrice ,
        sinon on retourne None.
        """
