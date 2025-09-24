#!/usr/bin/env python3
"""
Ce module définit une fonction permettant de calculer le coût total
d'un modèle TensorFlow en ajoutant la régularisation L2 au coût initial.
Il est utile pour entraîner des réseaux de neurones tout en limitant
le surapprentissage.
"""
import numpy as np
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calcule le coût total en ajoutant les pertes de régularisation L2
    d'un modèle TensorFlow au coût initial.

    La régularisation L2 permet de pénaliser les poids de grande amplitude
    afin de limiter le surapprentissage. Chaque couche du modèle qui
    définit une perte de régularisation (via `kernel_regularizer`) contribue
    à la valeur finale du coût.

    Args:
    cost (float ou tf.Tensor): Coût initial du modèle (par ex. cross-entropy).
    model (tf.keras.Model): Modèle TensorFlow entraîné, contenant
                            éventuellement des couches avec régularisation L2.

    Returns:
        tf.Tensor: Tableau contenant le coût total (coût + régularisation)
                   pour chaque couche avec régularisation.
    """
    total_costs = []

    for layer in (model.layers):
        if len(layer.losses) > 0:
            reg_loss = layer.losses[0]
            total_costs.append(reg_loss + cost)
    result = tf.stack(total_costs)
    return result
