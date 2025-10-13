#!/usr/bin/env python3
"""
Module implémentant un bloc de projection (projection block) pour
un réseau résiduel (ResNet) utilisant TensorFlow et Keras.

Ce bloc est une variante du bloc d'identité utilisée lorsque les
dimensions du tenseur d'entrée et de sortie diffèrent. Il introduit
une projection par convolution 1x1 dans le chemin de raccourci
(shortcut) afin d'adapter la taille et le nombre de canaux avant
l'addition résiduelle.
"""

from tensorflow import keras as K
    """
    Construit un bloc de projection pour un réseau résiduel (ResNet).

    Le bloc applique trois convolutions (1x1, 3x3, 1x1) avec des
    normalisations par batch et des activations ReLU. En parallèle,
    une convolution 1x1 (chemin de raccourci) projette l'entrée pour
    correspondre à la dimension de sortie avant l'addition résiduelle.

    Args:
        A_prev (keras.Tensor): tenseur d'entrée du bloc, de forme
            (batch_size, height, width, channels).
        filters (tuple or list): contient les dimensions des filtres
            pour chaque couche convolutive, dans l’ordre :
            (F11, F3, F12), où :
                - F11 : filtres de la première convolution 1x1
                - F3  : filtres de la convolution 3x3
                - F12 : filtres de la dernière convolution 1x1
        s (int, optional): facteur de stride pour la première
            convolution et la projection du raccourci. Défaut à 2.

    Returns:
        keras.Tensor: tenseur résultant après la connexion résiduelle
        et la dernière activation ReLU.

    Exemple:
        >> output = projection_block(A_prev, (64, 64, 256), s=2)
        >> print(output.shape)
    """

def projection_block(A_prev, filters, s=2):


    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(F11, (1, 1), strides=(s, s),
                    padding='same', kernel_initializer=init)(A_prev)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation("relu")(norm1)

    conv2 = K.layers.Conv2D(F3, (3, 3), (1, 1),
                    padding='same', kernel_initializer=init)(relu1)
    norm2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation("relu")(norm2)

    conv3 = K.layers.Conv2D(F12, (1, 1), (1, 1),
                    padding='same', kernel_initializer=init)(relu2)
    norm3 = K.layers.BatchNormalization(axis=3)(conv3)


    conv_shortcut = K.layers.Conv2D(F12, (1, 1), strides=(s, s),
                    padding='same', kernel_initializer=init)(A_prev)
    norm_shortcut = K.layers.BatchNormalization(axis=3)(conv_shortcut)
    shortcut = K.layers.Add()([norm3, norm_shortcut])

    return K.layers.Activation("relu")(shortcut)
