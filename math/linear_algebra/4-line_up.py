#!/usr/bin/env python3
"""
Cette fonction additionne deux tableaus élément par élément
"""


def add_arrays(arr1, arr2):
    """
    arr1 + arr2
    """

    if len(arr1) != len(arr2):
        return None
    """
    vérification des tailles
    """
    arr3 = [a + b for a, b in zip(arr1, arr2)]
    """
    créations de paire a et b pour chaque array et on les additiones (a + b)
    et cela position par position
    """
    return arr3
