#!/usr/bin/env python3
"""
Ce module définit une fonction utilitaire pour créer une couche Dense
TensorFlow/Keras avec régularisation Dropout. Elle permet de construire
des réseaux de neurones robustes en désactivant aléatoirement certains
neurones pendant l’entraînement pour limiter le surapprentissage.
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Crée une couche Dense avec Dropout pour un réseau TensorFlow/Keras.

    Args:
        prev (tf.Tensor): Entrée provenant de la couche précédente.
        n (int): Nombre de neurones dans la couche.
        activation (callable ou str): Fonction d’activation à appliquer.
        keep_prob (float): Probabilité de conserver un neurone actif
                           pendant le Dropout (0 < keep_prob ≤ 1).
        training (bool): Indique si le Dropout est actif (True pendant
                         l’entraînement, False en prédiction).

    Returns:
        tf.Tensor: Sortie de la couche Dense après application du Dropout
                   si `training=True`, sinon simple activation.
    """
    init_weights = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init_weights
    )

    output = layer(prev)

    if training:
        dropout = tf.keras.layers.Dropout(rate=(1 - keep_prob))
        output = dropout(output, training=training)

    return output
