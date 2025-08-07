#!/usr/bin/env python3
"""
Cette fonction sert à transposer une matrice
"""


def matrix_transpose(matrix):
    """
    transpose de matrix
    """
    rows = len(matrix)
    """
    pour les lignes
    """
    cols = len(matrix[0])
    """
    pour les collones
    """

    transposed_matrix = [[0 for index in range(rows)] for index in range(cols)]
    """
    créations de la matrice qui sera transposer
    """
    for i in range(rows):
        for j in range(cols):
            transposed_matrix[j][i] = matrix[i][j]
            """
            copie et transpose de la matrice en inversant les index i et j
            """
    return transposed_matrix
