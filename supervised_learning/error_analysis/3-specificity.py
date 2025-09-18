#!/usr/bin/env python3
"""
la spécificité, qui mesure la capacité d'un test à donner
un résultat négatif lorsque l'hypothèse n'est pas vérifiée.
"""
import numpy as np


def specificity(confusion):
    """
    Calcule la spécificité pour chaque classe à partir
    d'une matrice de confusion.

    La spécificité mesure la capacité d’un modèle à identifier correctement
    les exemples négatifs, c’est-à-dire la proportion de vrais négatifs (VN)
    parmi tous les cas réellement négatifs (VN + FP).

    Args:
    confusion (numpy.ndarray): Matrice de confusion
                               de taille (n_classes, n_classes).

    Returns:
        numpy.ndarray: Tableau 1D de taille (n_classes,),
        où chaque élément correspond à la spécificité de la classe i.

                       La formule utilisée est :
                           spécificité_i = VN_i / (VN_i + FP_i)
                       où :
                       - VN_i est le nombre de vrais négatifs pour la classe i
                        - FP_i est le nombre de faux positifs pour la classe i
    """
    classes = confusion.shape[0]
    specificity = np.zeros(classes)

    for i in range(classes):
        VN = confusion.sum() - confusion[i, :].sum()
        - confusion[:, i].sum() + confusion[i, i]
        FP = confusion[:, i].sum() - confusion[i, i]
        specificity[i] = VN / (VN + FP)
    return specificity
