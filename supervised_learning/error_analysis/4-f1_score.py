#!/usr/bin/env python3
"""
Le score F, ou mesure F, mesure la performance prédictive.
Il est calculé à partir de la précision et du rappel du test.
"""
import numpy as np

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calcule le F1-score pour chaque classe à partir d'une matrice de confusion.

    Le F1-score est la moyenne harmonique entre
    la précision et la sensibilité (rappel).
    Il permet d’équilibrer les deux mesures et est particulièrement utile
    en cas de déséquilibre entre classes.

    Args:
        confusion (numpy.ndarray):
        Matrice de confusion de taille (n_classes, n_classes).

    Returns:
        numpy.ndarray:
            Tableau 1D de taille (n_classes,), où chaque élément correspond
            au F1-score de la classe i.

    La formule utilisée est :
    F1_i = 2 * (précision_i * sensibilité_i) / (précision_i + sensibilité_i)
    """

    sens = sensitivity(confusion)
    prec = precision(confusion)

    f1 = 2 * (prec * sens) / (prec + sens)

    return f1
