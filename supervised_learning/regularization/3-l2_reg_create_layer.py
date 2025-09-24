#!/usr/bin/env python3
"""
Ce module définit une fonction utilitaire pour créer une couche Dense
de TensorFlow/Keras avec régularisation L2. Il facilite la construction
de réseaux de neurones tout en limitant le surapprentissage.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Crée une couche Dense TensorFlow avec initialisation des poids
    et régularisation L2.

    Args:
        prev (tf.Tensor): Entrée de la couche précédente.
        n (int): Nombre de neurones de la couche.
        activation (callable ou str): Fonction d'activation à appliquer.
        lambtha (float): Facteur de régularisation L2.

    Returns:
        tf.Tensor: La sortie de la couche créée appliquée sur l'entrée `prev`.
    """
    init_weights = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )
    L2_regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init_weights,
        kernel_regularizer=L2_regularizer
    )
    return layer(prev)
