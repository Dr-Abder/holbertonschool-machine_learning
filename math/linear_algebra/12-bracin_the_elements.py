#!/usr/bin/env python3
"""
Fonction arithmetique comprennent l'addition ( + ),
la soustraction ( - ),la multiplication ( * ), la division ( / ) 
"""


def np_elementwise(mat1, mat2):
    """
    Retourne mat1 et mat2 avec leur variable selon 
    leur type d'opération arithmetique
    """
    addition = mat1 + mat2
    subtraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2
    return (addition, subtraction, multiplication, division)
