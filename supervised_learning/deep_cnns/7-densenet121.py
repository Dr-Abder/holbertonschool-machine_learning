#!/usr/bin/env python3
"""
Module implémentant DenseNet-121 en utilisant TensorFlow Keras.

Ce module définit la fonction densenet121, qui construit le modèle
DenseNet-121 complet pour des images 224x224x3. Le réseau est composé
de blocs denses et de couches de transition pour réduire la taille
des cartes de caractéristiques et le nombre de filtres, suivi d'une
classification softmax sur 1000 classes.
"""

from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):

    """
Construit le modèle DenseNet-121.

Le réseau est composé de :
- Une couche d'entrée 224x224x3
- Une convolution initiale 7x7 suivie d'un max pooling
- 4 blocs denses séparés par 3 couches de transition
- Une pooling global moyen 7x7
- Une couche Dense softmax pour 1000 classes

Args:
growth_rate (int): nombre de filtres ajoutés par couche
        dans chaque bloc dense.
compression (float): facteur de réduction du nombre de filtres
        dans chaque couche de transition (0 < compression <= 1).

Returns:
keras.Model: modèle Keras DenseNet-121 prêt à l'entraînement.
"""


init = K.initializers.HeNormal(seed=0)

X_input = K.Input(shape=(224, 224, 3))

BN1 = K.layers.BatchNormalization(axis=3)(X_input)
RL1 = K.layers.Activation('relu')(BN1)
CNV1 = K.layers.Conv2D(growth_rate*2, (7, 7), (2, 2), padding='same',
                       kernel_initializer=init)(RL1)
MAXP1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(CNV1)

out_1, nb = dense_block(MAXP1, 64, growth_rate, 6)
trans_1, nb = transition_layer(out_1, nb, compression)

out_2, nb = dense_block(trans_1, nb, growth_rate, 12)
trans_2, nb = transition_layer(out_2, nb, compression)

out_3, nb = dense_block(trans_2, nb, growth_rate, 24)
trans_3, nb = transition_layer(out_3, nb, compression)

out_4, nb = dense_block(trans_3, nb, growth_rate, 16)

AVG = K.layers.AveragePooling2D(pool_size=7)(out_4)
X = K.layers.Dense(1000, activation='softmax')(AVG)

return K.models.Model(inputs=X_input, outputs=X)
