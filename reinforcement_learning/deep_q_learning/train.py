#!/usr/bin/env python3
"""
train.py
Entraîne un agent Deep Q-Network (DQN) sur Breakout d’Atari en utilisant keras-rl2 et Gymnasium.
La politique entraînée est sauvegardée sous le nom 'policy.h5'.
"""

import gymnasium as gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
from tensorflow.keras.optimizers.legacy import Adam  # Using legacy Adam optimizer for keras-rl2 compatibility
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


class CompatibilityWrapper(gym.Wrapper):
    """
    Un wrapper pour adapter l’environnement afin qu’il soit compatible avec keras-rl2.
    S’assure que l’environnement retourne les sorties attendues et gère les
    indicateurs de fin d’épisode conformément aux exigences de keras-rl2.
    """

    def step(self, action):
        """
        Effectue une étape dans l’environnement avec l’action donnée.

        :param action: L’action à effectuer dans l’environnement.
        :return: Tuple (observation, reward, done, info)
                 - observation : l’observation résultante
                 - reward : la récompense reçue pour cette étape
                 - done : indique si l’épisode est terminé
                 - info : informations supplémentaires concernant l’étape
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        """
        Réinitialise l’environnement et retourne l’observation initiale.

        :param kwargs: arguments supplémentaires pour la réinitialisation de l’environnement
        :return: observation initiale
        """
        observation, info = self.env.reset(**kwargs)
        return observation

def create_atari_environment(env_name):
    """
    Initialise et configure un environnement Atari pour l’entraînement.

    :param env_name: str, le nom de l’environnement Atari à initialiser
    :return: objet gym.Env configuré pour le jeu Atari
    """
    env = gym.make(env_name, render_mode='rgb_array')
    env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
    env = CompatibilityWrapper(env)
    return env


def build_model(window_length, shape, actions):
    """
    Construit un modèle de Réseau de Neurones Convolutif (CNN) pour l’apprentissage DQN.

    :param window_length: int, nombre d’images empilées en entrée pour représenter le mouvement
    :param shape: tuple, la forme des images individuelles (hauteur, largeur, canaux)
    :param actions: int, le nombre d’actions possibles dans l’environnement
    :return: modèle keras Sequential prêt pour l’entraînement DQN
    """
    model = Sequential()
    # Réorganiser les dimensions d’entrée pour respecter les exigences de keras-rl2
    model.add(Permute((2, 3, 1), input_shape=(window_length,) + shape))
    # Première couche convolutionnelle pour capturer les motifs spatiaux
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    # Deuxième couche convolutionnelle pour une reconnaissance plus profonde des motifs
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    # Troisième couche convolutionnelle pour affiner les caractéristiques spatiales
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    # Aplatir les cartes de caractéristiques en un seul vecteur
    model.add(Flatten())
    # Couche entièrement connectée pour traiter les caractéristiques combinées
    model.add(Dense(512, activation='relu'))
    # Couche de sortie avec activation linéaire pour prédire les valeurs Q de chaque action
    model.add(Dense(actions, activation='linear'))
    return model

# Configuration de l’agent DQN

if __name__ == "__main__":
    # Initialiser l’environnement
    env = create_atari_environment('ALE/Breakout-v5')
    nb_actions = env.action_space.n  # Nombre d’actions possibles dans Breakout

    # Construire et configurer le modèle
    window_length = 4  # Nombre d’images consécutives pour former une observation
    model = build_model(window_length, env.observation_space.shape, nb_actions)

    # Définir l’agent DQN avec une politique et un buffer mémoire
    memory = SequentialMemory(limit=1000000, window_length=window_length)  # Mémoire de relecture pour stocker les expériences passées
    policy = EpsGreedyQPolicy()  # Politique epsilon-greedy pour équilibrer exploration et exploitation
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        nb_steps_warmup=50000,  # Nombre d’étapes avant le début de l’entraînement pour remplir la mémoire
        gamma=0.99,  # Facteur d’actualisation des récompenses futures
        target_model_update=10000,  # Intervalle de mise à jour des poids du réseau cible
        train_interval=4,  # Fréquence des mises à jour de l’entraînement
        delta_clip=1.0  # Limitation du terme d’erreur pour la stabilité
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Entraîner l’agent DQN sur l’environnement
    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=2)

    # Sauvegarder les poids du modèle entraîné
    dqn.save_weights('policy.h5', overwrite=True)

    # Fermer l’environnement pour libérer les ressources
    env.close()
