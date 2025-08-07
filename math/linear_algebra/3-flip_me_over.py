#!/usr/bin/env python3
"""
Cette fonction sert à transposer une matrice
"""


def matrix_transpose(matrix):

    return [list(row) for row in zip(*matrix)]