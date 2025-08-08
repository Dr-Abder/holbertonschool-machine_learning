#!/usr/bin/env python3
"""
Cette fonction concatène arr1 et arr2 qui sont des listes
"""


def cat_arrays(arr1, arr2):
    """
    concatène arr1 et arr2
    """

    arr3 = list(arr1)
    arr3.extend(arr2)
    """
    copie de la list arr1 dans arr3
    et ensuite extension de arr3 avec la liste arr2
    seconde possibilité plus simple: return arr1 + arr2
    """
    return arr3
