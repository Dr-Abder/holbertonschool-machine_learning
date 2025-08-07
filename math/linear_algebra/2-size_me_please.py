#!/usr/bin/env python3

def matrix_shape(matrix):
    """récursivité"""
    if not isinstance(matrix, list):
        return []
        """si matrix n'est pas une liste"""
    if isinstance(matrix[0], list):
        return [len(matrix)] + matrix_shape(matrix[0])
        """pour une matrice 3D (ou plus)"""
    else:
        return [len(matrix)]
        """pour une matrice 2D"""
