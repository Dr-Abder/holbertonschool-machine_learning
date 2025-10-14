#!/usr/bin/env python3


from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):

    init = K.initializers.HeNormal(seed=0)

    new_nbf = int(nb_filters * compression)

    BN1 = K.layers.BatchNormalization(axis=3)(X)
    RL1 = K.layers.Activation('relu')(BN1)
    CNV1 = K.layers.Conv2D(new_nbf, (1,1), padding='same',
                           kernel_initializer=init)(RL1)
    AVG = K.layers.AveragePooling2D((2, 2), padding='same',)(CNV1)

    return AVG, new_nbf
