#!/usr/bin/env python3
"""
Matrice de confusion pour visualiser les résultats
d'un algorithme de classification.
"""

import numpy as np


def create_confusion_matrix(labels, logits):

    """
    Crée une matrice de confusion à partir
    des étiquettes réelles et des prédictions.

    Args:
        labels (numpy.ndarray): Tableau 2D des étiquettes réelles
                                encodées en one-hot
                                (shape: [n_samples, n_classes]).
        logits (numpy.ndarray): Tableau 2D des prédictions du modèle
                                (shape: [n_samples, n_classes]).

    Returns:
        numpy.ndarray: Matrice de confusion de taille (n_classes, n_classes),
                       où l’élément [i, j] correspond au nombre de fois où
                       la classe i a été prédite comme j.
    """

    classes = labels.shape[1]

    confusion_matrix = np.zeros((classes, classes))

    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    for i in range(len(true_labels)):

        true_classe = true_labels[i]
        pred_classe = pred_labels[i]

        confusion_matrix[true_classe, pred_classe] += 1

    return confusion_matrix
