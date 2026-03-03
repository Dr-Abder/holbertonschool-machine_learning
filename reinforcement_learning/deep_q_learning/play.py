#!/usr/bin/env python3
"""
Visualise un agent Deep Q-Learning (DQN) jouant à Breakout d’Atari
à l’aide d’une politique entraînée. L’environnement du jeu est affiché avec Pygame,
et les performances de l’agent sont évaluées sur plusieurs épisodes.
"""

from __future__ import division

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Permute
import time
import pygame
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from rl.util import *
from rl.core import Processor
from rl.callbacks import Callback


class CompatibilityWrapper(gym.Wrapper):
    """
    Wrapper permettant d’assurer la compatibilité avec les anciennes versions de Gymnasium.
    Modifie les méthodes step et reset afin de fournir des sorties cohérentes.

    Attributs :
        env (gym.Env) : l’environnement Gym encapsulé.
    """

    def step(self, action):
        """
        Exécute une action donnée dans l’environnement.

        Args:
            action (int) : L’action à effectuer.

        Returns:
            tuple : Contient les éléments suivants :
                - observation (object) : La prochaine observation de l’environnement.
                - reward (float) : La récompense reçue après l’exécution de l’action.
                - done (bool) : True si l’épisode est terminé, sinon False.
                - info (dict) : Informations supplémentaires spécifiques à l’environnement.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        """
        Réinitialise l’environnement à son état initial.

        Args:
            **kwargs : Arguments supplémentaires pour la méthode reset de l’environnement.

        Returns:
            observation (object) : L’observation initiale de l’environnement.
        """
        observation, info = self.env.reset(**kwargs)
        return observation


def create_atari_environment(env_name):
    """
    Configure et prétraite un environnement Atari pour l’apprentissage par renforcement.

    Args:
        env_name (str) : Le nom de l’environnement Atari.

    Returns:
        gym.Env : L’environnement configuré.
    """
    env = gym.make(env_name, render_mode='rgb_array')

    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=1,
        noop_max=30
    )

    env = CompatibilityWrapper(env)
    return env


def build_model(window_length, shape, actions):
    """
    Construit un modèle de Réseau de Neurones Convolutif (CNN) pour traiter
    des images empilées dans l’environnement Atari.

    Args:
        window_length (int) : Le nombre d’images à empiler en entrée.
        shape (tuple) : La forme d’une image d’entrée (hauteur, largeur, canaux).
        actions (int) : Le nombre total d’actions possibles dans l’environnement.

    Returns:
        keras.models.Sequential : Le modèle CNN construit.
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(window_length,) + shape))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


class AtariProcessor(Processor): 
    """
    Processeur personnalisé pour le prétraitement des observations, des récompenses
    et des lots d’états dans l’environnement Atari avant de les transmettre à l’agent DQN.
    """

    def process_observation(self, observation):
        """
        Convertit les observations dans un format cohérent (tableau NumPy).

        Args:
            observation (object): L’observation brute provenant de l’environnement.

        Returns:
            np.ndarray: Observation traitée.
        """
        if isinstance(observation, tuple):
            observation = observation[0]
        img = np.array(observation, dtype='uint8')
        return img

    def process_state_batch(self, batch):
        """
        Normalise les valeurs des pixels d’un lot d’états dans l’intervalle [0, 1].

        Args:
            batch (np.ndarray): Lot d’états.

        Returns:
            np.ndarray: Lot d’états normalisé.
        """
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """
        Tronque les récompenses pour les maintenir dans l’intervalle [-1, 1].

        Args:
            reward (float): Récompense brute provenant de l’environnement.

        Returns:
            float: Récompense tronquée.
        """
        return np.clip(reward, -1., 1.)


# CALLBACK D’AFFICHAGE PYGAME

class PygameCallback(Callback):
    """
    Callback pour visualiser le jeu de l’agent en utilisant Pygame.
    """

    def __init__(self, env, delay=0.02):
        """
        Initialise le PygameCallback.

        Args:
            env (gym.Env): L’environnement à visualiser.
            delay (float): Temps (en secondes) de pause entre les images affichées.
        """
        self.env = env
        self.delay = delay
        pygame.init()
        self.screen = pygame.display.set_mode((420, 320))
        pygame.display.set_caption("Atari Breakout - DQN Agent")

    def on_action_end(self, action, logs={}):
        """
        S’exécute après une action de l’agent et affiche l’image avec Pygame.

        Args:
            action (int): L’action effectuée par l’agent.
            logs (dict): Journaux liés à l’entraînement (optionnel).
        """
        # Afficher l’image actuelle de l’environnement
        frame = self.env.render()
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (420, 320))
        self.screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Gérer les événements Pygame (ex : fermeture de la fenêtre)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
                pygame.quit()

        time.sleep(self.delay)

    def on_episode_end(self, episode, logs={}):
        """
        S’exécute à la fin d’un épisode et applique une courte pause.

        Args:
            episode (int): Le numéro de l’épisode qui vient de se terminer.
            logs (dict): Journaux liés à l’entraînement (optionnel).
        """
        pygame.time.wait(1000)


# SCRIPT PRINCIPAL

if __name__ == "__main__":
    # Créer l’environnement Atari Breakout
    env = create_atari_environment('ALE/Breakout-v5')
    nb_actions = env.action_space.n

    # Construire le modèle CNN pour l’agent
    window_length = 4
    input_shape = (84, 84)
    model = build_model(window_length, input_shape, nb_actions)

    # Charger les poids du modèle pré-entraîné
    model.load_weights('policy.h5')

    # Configurer l’agent DQN
    memory = SequentialMemory(limit=1000000, window_length=window_length)
    processor = AtariProcessor()
    policy = GreedyQPolicy()
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )
    dqn.compile(optimizer='adam', metrics=['mae'])

    # Tester les performances de l’agent et visualiser le jeu
    pygame_callback = PygameCallback(env, delay=0.02)
    scores = dqn.test(env, nb_episodes=5, visualize=False, callbacks=[pygame_callback])

    # Afficher le score moyen
    print('Average score over 5 test episodes:', np.mean(scores.history['episode_reward']))

    # Fermer l’environnement et Pygame
    env.close()
    pygame.quit()
