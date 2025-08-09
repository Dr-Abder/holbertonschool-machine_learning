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
        blablba
        """

    if axis == 1:
        if len(mat1) == len(mat2):
            return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
        else:
            return None
        """
        bloblo
        """
