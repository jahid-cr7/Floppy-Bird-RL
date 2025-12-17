"""
Deep Q-Learning for Flappy Bird

This package implements a Deep Q-Network (DQN) reinforcement learning agent
for playing the Flappy Bird game using PyTorch.

@author: Zonaid Hossain, Abrar Bin Karim, Jahid Hasan
"""

__version__ = "2.0.0"

from .deep_q_network import DeepQNetwork
from .flappy_bird import FlappyBird
from .utils import preprocess_frame, stack_frames

__all__ = [
    "DeepQNetwork",
    "FlappyBird",
    "preprocess_frame",
    "stack_frames",
]

