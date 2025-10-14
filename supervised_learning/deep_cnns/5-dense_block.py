#!/usr/bin/env python3


from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):

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