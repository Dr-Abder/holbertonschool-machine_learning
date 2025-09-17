#!/usr/bin/env python3
"""
La spécificité (ou précision positive par classe) d’un test mesure
la capacité à donner un résultat négatif correct lorsque l’hypothèse
n’est pas vérifiée.
"""

import numpy as np


def precision(confusion):
    """
    Calcule la spécificité (ou précision positive) pour chaque classe
    à partir d'une matrice de confusion.

    La spécificité mesure la proportion de vrais négatifs correctement détectés
    ,ou de manière équivalente ici, la proportion de prédictions correctes pour
    une classe donnée parmi toutes les prédictions faites pour cette classe.

    Args:
        confusion (numpy.ndarray): Matrice de confusion
                                   de taille (n_classes, n_classes).

    Returns:
        numpy.ndarray: Tableau 1D de taille (n_classes,),
                       où chaque élément correspond à la spécificité
                       (ou précision positive) de la classe i.

                       La formule utilisée est :
                           spécificité_i = VP_i / (VP_i + FP_i)
                       où :
                       - VP_i est le nombre de vrais positifs pour la classe i
                        - FP_i est le nombre de faux positifs pour la classe i
    """
    classes = confusion.shape[1]
    specificity = np.zeros(classes)

    for i in range(len(specificity)):
        VP = confusion[i, i]
        total_V = confusion[:, i].sum()
        specificity[i] = VP / total_V
    return specificity
