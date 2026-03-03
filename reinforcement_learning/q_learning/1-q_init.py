#!/usr/bin/env python3


import numpy as np


def q_init(env):

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q_table = np.zeros((num_states, num_actions))
    return Q_table
