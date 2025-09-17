#!/usr/bin/env python3
"""
la sensibilité (ou sélectivité) d'un test mesure sacapacité
à donner un résultat positif lorsqu'une hypothèse est vérifiée.
"""
import numpy as np


def sensitivity(confusion):

    """
    Calcule la sensibilité (ou rappel) pour chaque
    classe à partir d'une matrice de confusion.

    La sensibilité mesure la proportion de vrais positifs correctement détectés
    parmi tous les exemples réellement positifs d'une classe donnée.

    Args:
        confusion (numpy.ndarray): Matrice de confusion
                                   de taille (n_classes, n_classes).

    Returns:
        numpy.ndarray: Tableau 1D de taille (n_classes,),
        où chaque élément correspond à la sensibilité de la classe i.
        La sensibilité est calculée comme :
                        sensibilité_i = VP_i / (VP_i + FN_i)
                        où VP_i est le nombre de vrais positifs de la classe i,
                        et FN_i le nombre de faux négatifs.
    """

    classes = confusion.shape[1]
    sensitivity = np.zeros(classes)

    for i in range(len(sensitivity)):
        VP = confusion[i, i]
        total_V = confusion[i].sum()
        sensitivity[i] = VP / total_V
    return sensitivity
