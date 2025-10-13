#!/usr/bin/env python3
"""
Module implémentant un bloc d'identité (identity block) pour ResNet
utilisant TensorFlow et Keras.

Ce bloc est utilisé dans les réseaux résiduels pour permettre la
propagation directe de l'information via une connexion résiduelle.
Il applique une succession de convolutions (1x1, 3x3, 1x1), de
normalisations par batch et de fonctions d'activation ReLU, puis
ajoute l'entrée originale pour former un chemin de gradient direct.
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):

    """
    Construit un bloc d'identité pour un réseau résiduel (ResNet).

    Ce bloc permet à l'entrée du bloc d'être additionnée à la sortie
    après une série de convolutions et de normalisations, ce qui
    facilite l'entraînement de réseaux profonds en préservant le
    gradient.

    Args:
        A_prev (keras.Tensor): tenseur d'entrée du bloc, de forme
            (batch_size, height, width, channels).
        filters (tuple or list): dimensions des filtres pour chaque
            convolution, dans l'ordre :
            (F11, F3, F12) où :
                - F11 : filtres de la première convolution 1x1
                - F3  : filtres de la convolution 3x3
                - F12 : filtres de la dernière convolution 1x1

    Returns:
        keras.Tensor: le tenseur de sortie après la connexion résiduelle
        et la dernière activation ReLU.

    Exemple:
        >> output = identity_block(A_prev, (64, 64, 256))
        >> print(output.shape)
    """

    F11, F3, F12 = filters

    init = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(F11, (1,1), padding='same',
                        kernel_initializer=init)(A_prev)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation("relu")(norm1)

    conv2 = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=init)(relu1)
    norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation("relu")(norm2)

    conv3 = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=init)(relu2)
    norm3 = K.layers.BatchNormalization(axis=3)(conv3)
    merge = K.layers.Add()([norm3, A_prev])

    return K.layers.Activation("relu")(merge)