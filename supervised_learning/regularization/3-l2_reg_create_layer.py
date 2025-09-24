#!/usr/bin/env python3

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):

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
