#!/usr/bin/env python3
"""
Module principal pour l'entraînement d'un modèle MobileNetV2
sur le dataset CIFAR-10 en utilisant le transfert learning.

Ce module :
- Charge et prétraite les données CIFAR-10.
- Redimensionne les images pour MobileNetV2.
- Crée un modèle CNN avec MobileNetV2 comme base pré-entraînée.
- Effectue un entraînement initial avec la base gelée.
- Débloque partiellement la base pour fine-tuning.
- Sauvegarde le modèle final au format HDF5.
"""

from tensorflow import keras as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Prétraite les données CIFAR-10.

    Paramètres
    ----------
    X : numpy.ndarray
        Images à prétraiter.
    Y : numpy.ndarray
        Labels à encoder.

    Retour
    ------
    tuple
        X prétraité pour MobileNetV2, Y encodé en one-hot sur 10 classes.
    """
    # Prétraitement des images selon MobileNetV2
    X_p = preprocess_input(X)
    # Encodage des labels en one-hot
    Y_p = to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    # Chargement des données CIFAR-10
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # Prétraitement des données
    X_train, Y_train = preprocess_data(x_train, y_train)
    X_test, Y_test = preprocess_data(x_test, y_test)

    # Couche Lambda pour redimensionner les images de 32x32 à 96x96
    R_layer = K.layers.Lambda(
        lambda x: tf.image.resize(x, (96, 96)),
        output_shape=(96, 96, 3)
    )

    # Chargement de MobileNetV2 pré-entraîné sans la tête de classification
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(96, 96, 3)
    )
    # On gèle les poids pour l'entraînement initial
    base_model.trainable = False

    data_augmentation = K.Sequential([
        K.layers.RandomFlip("horizontal"),   # flip horizontal aléatoire
        K.layers.RandomRotation(0.1),       # rotation légère
        K.layers.RandomZoom(0.1),            # zoom aléatoire
        K.layers.RandomContrast(0.1)
    ])

    # Création de l'input layer du modèle
    inputs = K.Input(shape=(32, 32, 3))
    X = data_augmentation(inputs)  # appliquer l'augmentation
    X = R_layer(X)      # Redimensionnement
    X = base_model(X)        # Passage par la base MobileNetV2

    # Pooling global et couche de sortie
    X = K.layers.GlobalAveragePooling2D()(X)
    X = Dense(256, activation='relu')(X)
    X = K.layers.Dropout(0.3)(X)  # régularisation
    X = Dense(10, activation='softmax')(X)

    # Création du modèle final
    model = Model(inputs=inputs, outputs=X)

    # Compilation du modèle pour entraînement initial
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        K.callbacks.EarlyStopping(
            monitor='val_loss',  # surveille la perte sur validation
            patience=3,          # stop si aucune amélioration après 3 epochs
            restore_best_weights=True  # restaurer le meilleur modèle
        ),
        K.callbacks.ModelCheckpoint(
            "best_model.h5",     # fichier où sauvegarder
            monitor='val_loss',
            save_best_only=True
        )
    ]

    # Entraînement initial avec base gelée
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=64,
        epochs=15,
        callbacks=callbacks
    )

    # Déblocage partiel de la base pour fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:125]:
        layer.trainable = False

    # Compilation pour le fine-tuning avec un learning rate réduit
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fine-tuning
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        batch_size=64,
        epochs=15,
        callbacks=callbacks
    )

    # Sauvegarde du modèle final
    model.save('cifar10.h5')
