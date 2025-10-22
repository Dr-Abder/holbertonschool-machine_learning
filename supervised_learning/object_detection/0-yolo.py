#!/usr/bin/env python3
"""
Module définissant la classe Yolo pour le traitement et la détection
d’objets à l’aide du modèle YOLO (You Only Look Once).

Ce module permet de charger un modèle YOLO pré-entraîné, ses classes
associées, ainsi que les paramètres nécessaires à la détection tels que
les seuils de confiance et les ancres utilisées pour le découpage des
zones de détection.
"""

from tensorflow import keras as K
import numpy as np


class Yolo:
    """
    Classe représentant le modèle YOLO pour la détection d’objets.

    Cette classe gère le chargement du modèle, des classes et des
    paramètres de configuration (seuils et ancres) utilisés lors de
    l’inférence.

    Attributes:
        model (keras.Model): modèle YOLO pré-entraîné chargé depuis
            un fichier.
        class_names (list): liste des noms de classes détectables.
        class_t (float): seuil minimal de confiance pour conserver
            une détection.
        nms_t (float): seuil de suppression non maximale (NMS) pour
            éliminer les détections redondantes.
        anchors (numpy.ndarray): tableau des ancres définissant les
            tailles et ratios des boîtes de détection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialise une instance de la classe Yolo.

        Charge le modèle YOLO pré-entraîné, les classes, et configure
        les seuils de confiance ainsi que les ancres nécessaires pour
        la détection d’objets.

        Args:
            model_path (str): chemin vers le fichier du modèle YOLO.
            classes_path (str): chemin vers le fichier contenant la
                liste des classes (une par ligne).
            class_t (float): seuil de confiance minimal pour valider
                une détection.
            nms_t (float): seuil utilisé pour la suppression non
                maximale (NMS).
            anchors (numpy.ndarray): tableau contenant les ancres.
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = f.read().strip().split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
