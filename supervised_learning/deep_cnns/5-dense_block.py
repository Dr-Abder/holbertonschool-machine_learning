#!/usr/bin/env python3
"""
Construit un bloc dense tel que défini dans l’architecture DenseNet.
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):

    """
    Construit un bloc dense pour un réseau DenseNet.

    Un bloc dense est une séquence de couches convolutionnelles dans laquelle
    chaque couche reçoit en entrée les cartes de caractéristiques
    (feature maps) de toutes les couches précédentes du même bloc.
    Ce type de connexion favorise la réutilisation des caractéristiques,
    améliore la propagation du gradient et réduit le nombre de paramètres
    nécessaires pour l’entraînement.

    À l’intérieur du bloc, chaque couche effectue les opérations suivantes :
        1. Normalisation par lot (Batch Normalization)
        2. Activation ReLU
        3. Convolution 1x1 (couche de réduction, 4 * growth_rate filtres)
        4. Normalisation par lot
        5. Activation ReLU
        6. Convolution 3x3 (growth_rate filtres)
        7. Concaténation avec les entrées précédentes

    Arguments :
        X (tensor) : tenseur d’entrée du bloc.
        nb_filters (int) : nombre de filtres présents dans X avant le bloc.
        growth_rate (int) : taux de croissance du bloc dense,
                        c’est-à-dire le nombre de filtres ajoutés par couche.
        layers (int) : nombre de couches à empiler dans le bloc dense.

    Retourne :
        tuple :
            -X_concat (tensor): sortie du bloc après concaténation des couches.
            -nb_filters (int) : nombre total de filtres après le bloc dense
                                 (nb_filters + growth_rate * layers).
    """
    init = K.initializers.HeNormal(seed=0)
    for i in range(layers):

        BN1 = K.layers.BatchNormalization(axis=3)(X)
        RL1 = K.layers.Activation('relu')(BN1)
        CNV1 = K.layers.Conv2D(4*growth_rate, (1, 1), padding='same',
                               kernel_initializer=init)(RL1)

        BN2 = K.layers.BatchNormalization(axis=3)(CNV1)
        RL2 = K.layers.Activation('relu')(BN2)
        CNV2 = K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                               kernel_initializer=init)(RL2)

        X_concat = K.layers.Concatenate(axis=3)([X, CNV2])
        X = X_concat
        nb_filters = nb_filters + growth_rate

    return X_concat, nb_filters
