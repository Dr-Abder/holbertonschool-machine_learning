#!/usr/bin/env python3
"""
Implémente l’architecture du réseau de neurones LeNet-5 avec Keras.

Ce module définit la fonction `lenet5` qui construit et compile un modèle
LeNet-5 adapté à la classification d’images. L’architecture comporte deux
couches de convolution, deux couches de pooling, puis trois couches
complètement connectées.
"""

from tensorflow import keras as K


def lenet5(X):
    """
    Construit et compile le modèle LeNet-5 à l’aide de Keras.

    Parameters
    ----------
    X : keras.Input
        Entrée du réseau, généralement de forme (m, 28, 28, 1)
        pour des images MNIST en niveaux de gris.

    Returns
    -------
    keras.Model
        Le modèle LeNet-5 compilé avec l’optimiseur Adam, la
        fonction de perte 'categorical_crossentropy' et la
        métrique 'accuracy'.
    """
    initializer = K.initializers.HeNormal(seed=0)

    # C1 : Convolution 6 filtres 5x5, padding same, ReLU
    conv1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=initializer
    )(X)

    # S2 : Max pooling 2x2, stride 2
    pool1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)

    # C3 : Convolution 16 filtres 5x5, padding valid, ReLU
    conv2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=initializer
    )(pool1)

    # S4 : Max pooling 2x2, stride 2
    pool2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)

    # Flatten avant les couches fully connected
    flatten = K.layers.Flatten()(pool2)

    # C5 : Fully connected 120 neurones, ReLU
    fc1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=initializer
    )(flatten)

    # F6 : Fully connected 84 neurones, ReLU
    fc2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=initializer
    )(fc1)

    # OUTPUT : Fully connected 10 neurones, Softmax
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=initializer
    )(fc2)

    model = K.Model(inputs=X, outputs=output)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
