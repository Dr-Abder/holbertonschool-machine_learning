#!/usr/bin/env python3
"""
Module permettant à l’agent entraîné de jouer un épisode sur FrozenLake.
"""

import numpy as np


def play(env, Q, max_steps=100):
    """
    Joue un épisode en utilisant la Q-table entraînée.
    """
    state = env.reset()[0]
    rendered_outputs = []
    total_rewards = 0

    for _ in range(max_steps):
        # Afficher et capturer l'état actuel de l'environnement
        rendered_outputs.append(env.render())

        # Choisir la meilleure action (exploitation de la Q-table)
        action = np.argmax(Q[state])

        # Exécuter l'action
        next_state, reward, done, _, _ = env.step(action)

        # Mettre à jour la récompense totale
        total_rewards += reward

        # Passer à l'état suivant
        state = next_state

        # Terminer l'épisode si c'est fini
        if done:
            break

    # S'assurer que l'état final est également affiché après la fin de l'épisode
    rendered_outputs.append(env.render())

    return total_rewards, rendered_outputs
