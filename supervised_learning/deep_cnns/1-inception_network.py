#!/usr/bin/env python3
"""
Module implémentant le réseau Inception (GoogLeNet) en utilisant
TensorFlow et Keras.

Ce module définit la fonction inception_network qui construit une
architecture complète du réseau de neurones convolutif Inception v1.
Le modèle combine plusieurs blocs Inception pour extraire des
caractéristiques à différentes échelles, ce qui permet d’améliorer
les performances de classification tout en limitant le coût
computationnel grâce aux convolutions 1x1.
"""

from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():

    """
    Construit le modèle complet du réseau Inception (GoogLeNet).

    Le réseau est constitué d’une succession de couches convolutives,
    de couches de pooling et de blocs Inception. Chaque bloc combine
    plusieurs convolutions (1x1, 3x3, 5x5) et un max pooling afin
    d’extraire des caractéristiques locales et globales simultanément.

    L’architecture finale se termine par une couche de pooling
    global moyen, suivie d’un dropout et d’une couche dense de sortie
    de 1000 neurones avec une activation softmax, adaptée à la
    classification sur 1000 classes (ImageNet).

    Returns:
        keras.Model: le modèle Keras correspondant à l’architecture
        complète du réseau Inception v1 (GoogLeNet).

    Exemple:
        >> model = inception_network()
        >> model.summary()
    """

        X = K.Input(shape=(224, 224, 3))

        conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                                padding='same', activation='relu')(X)
        max_pool1 = K.layers.MaxPooling2D(
                (3, 3), strides=(2, 2), padding='same')(conv1)    
        conv2_reduce = K.layers.Conv2D((64), (1, 1),
                                padding='same', activation='relu')(max_pool1)
        conv2 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),
                                padding='same', activation='relu')(conv2_reduce)
        max_pool2 = K.layers.MaxPooling2D(
                (3, 3), (2, 2), padding='same')(conv2)

        incept3a = inception_block(max_pool2, [64, 96, 128, 16, 32, 32,])
        incept3b = inception_block(incept3a, [128, 128, 192, 32, 96, 64])

        max_pool3 = K.layers.MaxPooling2D(
                (3, 3), (2, 2), padding='same')(incept3b)

        incept4a = inception_block(max_pool3, [192, 96, 208, 16, 32, 32])
        incept4b = inception_block(incept4a, [160, 112, 224, 24, 64, 64])
        incept4c = inception_block(incept4b, [128, 128, 256, 24, 64, 64])
        incept4d = inception_block(incept4c, [112, 144, 288, 32, 64, 64])
        incept4e = inception_block(incept4d, [256, 160, 320, 32, 128, 128])

        max_pool4 = K.layers.MaxPooling2D(
                (3, 3), (2, 2), padding='same')(incept4e)
        
        incept5a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])
        incept5b = inception_block(incept5a, [384, 192, 384, 48, 128, 128])

        avg_pool = K.layers.AveragePooling2D(
                (7, 7), (1, 1), padding='same')(incept5b)
        
        dropout = K.layers.Dropout(0.4)(avg_pool)
        output = K.layers.Dense(1000, activation='softmax')(dropout)
        model = K.models.Model(inputs=X, outputs=output)
        return model
