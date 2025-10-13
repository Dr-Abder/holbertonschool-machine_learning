#!/usr/bin/env python3
"""
Module implémentant un bloc Inception pour un réseau de neurones
convolutif utilisant TensorFlow et Keras.

Ce module définit la fonction inception_block qui construit une
architecture inspirée du modèle GoogLeNet. Chaque bloc combine
plusieurs convolutions de tailles différentes et un pooling afin
de capturer simultanément des caractéristiques locales et globales.
"""

from tensorflow import keras as K


def inception_block(A_prev, filters):

    """
    Construit un bloc Inception à partir d’un tenseur d’entrée.

    Le bloc Inception permet d’appliquer en parallèle plusieurs
    convolutions de tailles différentes (1x1, 3x3, 5x5) ainsi qu’un
    max pooling suivi d’une convolution 1x1. Les sorties de chaque
    branche sont ensuite concaténées sur l’axe des canaux.

    Args:
        A_prev (keras.Tensor): la sortie du bloc précédent ou
            l’entrée du réseau (un tenseur 4D de forme
            (batch_size, height, width, channels)).
        filters (tuple or list): contient les dimensions des filtres
            à utiliser pour chaque branche du bloc, dans l’ordre :
            (F1, F3R, F3, F5R, F5, FPP)
            où :
                - F1  : filtres de la convolution 1x1
                - F3R : filtres de réduction avant la convolution 3x3
                - F3  : filtres de la convolution 3x3
                - F5R : filtres de réduction avant la convolution 5x5
                - F5  : filtres de la convolution 5x5
                - FPP : filtres de la convolution suivant le max pooling

    Returns:
        keras.Tensor: le tenseur résultant de la concaténation des
        différentes branches du bloc Inception.

    Exemple:
        >> output = inception_block(A_prev, (64, 96, 128, 16, 32, 32))
        >> print(output.shape)
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    conv_1x1 = K.layers.Conv2D(
        F1, (1, 1), padding="same", activation='relu')(A_prev)

    conv_3x3_reduce = K.layers.Conv2D(
        F3R, (1, 1), padding="same", activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2D(
        F3, (3, 3), padding="same", activation='relu')(conv_3x3_reduce)

    conv_5x5_reduce = K.layers.Conv2D(
        F5R, (1, 1), padding="same", activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2D(
        F5, (5, 5), padding="same", activation='relu')(conv_5x5_reduce)

    max_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(1, 1), padding="same")(A_prev)
    conv_max_pool = K.layers.Conv2D(
        FPP, (1, 1), padding="same", activation='relu')(max_pool)

    output = K.layers.Concatenate(axis=3)([conv_1x1, conv_3x3,
                                           conv_5x5, conv_max_pool])
    return output
