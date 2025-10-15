#!/usr/bin/env python3
"""
Module définissant la fonction transition_layer, utilisée dans la
construction d’un réseau DenseNet. Elle ajoute une couche de transition
entre deux blocs de convolution pour réduire la dimension spatiale et
le nombre de filtres, améliorant ainsi la compacité du modèle.
"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):

    """
    Crée une couche de transition pour un DenseNet.

    Cette couche applique une normalisation par lot, une activation ReLU,
    une convolution 1x1 suivie d’un pooling moyen 2x2. Elle permet de
    réduire le nombre de filtres et la taille spatiale des cartes de
    caractéristiques, limitant la complexité du réseau.

    Args:
        X (keras.Tensor): tenseur de sortie de la couche précédente.
        nb_filters (int): nombre de filtres en entrée.
        compression (float): facteur de réduction du nombre de filtres
            (compris entre 0 et 1).

    Returns:
        tuple: (Y, new_nbf)
            - Y (keras.Tensor): sortie de la couche de transition.
            - new_nbf (int): nouveau nombre de filtres après compression.
    """
    init = K.initializers.HeNormal(seed=0)

    new_nbf = int(nb_filters * compression)

    BN1 = K.layers.BatchNormalization(axis=3)(X)
    RL1 = K.layers.Activation('relu')(BN1)
    CNV1 = K.layers.Conv2D(new_nbf, (1, 1), padding='same',
                           kernel_initializer=init)(RL1)
    AVG = K.layers.AveragePooling2D((2, 2), padding='same',)(CNV1)

    return AVG, new_nbf
