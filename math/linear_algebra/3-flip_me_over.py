#!/usr/bin/env python3
"""
Cette fonction sert à transposer une matrice 2D
"""


def matrix_transpose(matrix):
    """
    transpose de matrix
    """
    return [list(row) for row in zip(*matrix)]