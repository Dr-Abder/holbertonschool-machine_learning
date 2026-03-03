#!/usr/bin/env python3


import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):

    env = gym.make("FrozenLake-v1",
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery,
                   render_mode="ansi")  # Add render_mode here
    return env
