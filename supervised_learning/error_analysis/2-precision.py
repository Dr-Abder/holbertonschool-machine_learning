#!/usr/bin/env python3
"""
La précision (ou *precision*) d’un test mesure la capacité
d’un modèle à faire des prédictions correctes pour une classe donnée
parmi toutes les prédictions faites pour cette classe.
"""

import numpy as np


def precision(confusion):
    """
    Calcule la précision pour chaque classe à
    partir d'une matrice de confusion.

    La précision mesure la proportion de vrais positifs correctement prédits
    parmi toutes les prédictions faites pour une classe donnée.

    Args:
        confusion (numpy.ndarray): Matrice de confusion
                                   de taille (n_classes, n_classes).

    Returns:
        numpy.ndarray:
            Tableau 1D de taille (n_classes,),
            où chaque élément correspond à la précision de la classe i.

                   La formule utilisée est :
                        précision_i = VP_i / (VP_i + FP_i)
                   où :
                       - VP_i est le nombre de vrais positifs pour la classe i
                       - FP_i est le nombre de faux positifs pour la classe i
    """
    classes = confusion.shape[1]
    precision = np.zeros(classes)

    for i in range(len(precision)):
        VP = confusion[i, i]
        total_VP_FP = confusion[:, i].sum()
        precision[i] = VP / total_VP_FP
    return precision
