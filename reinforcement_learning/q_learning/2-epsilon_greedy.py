#!/usr/bin/env python3


import numpy as np


def epsilon_greedy(Q, state, epsilon):

    # Decide whether to explore or exploit
    if np.random.uniform(0, 1) < epsilon:
        # Exploration: choose a random action
        action = np.random.randint(Q.shape[1])
    else:
        # Exploitation: choose the best action from Q-table
        action = np.argmax(Q[state])

    return action
