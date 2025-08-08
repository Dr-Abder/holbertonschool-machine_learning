#!/usr/bin/env python3
"""
Cette fonction additionne deux matrices élément par élément
"""


def add_matrices2D(mat1, mat2):
    """
    mat1 + mat2
    """

    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
    """
    vérification si les matrices on les mêmes longueurs (lignes et colonnes)
    """
    mat3 = [[sum(values) for values in zip(*rows)] for rows in zip(mat1, mat2)]
    """
    addition dans une variable qui itère de tuple en tuple
    en aditionant chaque valeur qui sont ranger par paire au même indice
    """
    return mat3
