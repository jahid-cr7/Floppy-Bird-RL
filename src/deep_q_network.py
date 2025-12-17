"""
Deep Q-Network (DQN) Architecture Implementation

This module implements a convolutional neural network architecture for Deep Q-Learning,
specifically designed for processing game frames and learning optimal action policies
in reinforcement learning scenarios.

@author: Zonaid Hossain, Abrar Bin Karim, Jahid Hasan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DeepQNetwork(nn.Module):
    """
    Deep Q-Network architecture for reinforcement learning.
    
    This network processes sequences of game frames (4 stacked frames) and outputs
    Q-values for each possible action. The architecture consists of three convolutional
    layers followed by two fully connected layers, designed to extract spatial features
    from game screens and learn action-value mappings.
    
    Attributes:
        conv1: First convolutional layer (4 channels -> 32 channels)
        conv2: Second convolutional layer (32 channels -> 64 channels)
        conv3: Third convolutional layer (64 channels -> 64 channels)
        fc1: First fully connected layer (feature extraction)
        fc2: Output layer producing Q-values for each action
    """
    
    def __init__(self, num_actions: int = 2) -> None:
        """
        Initialize the Deep Q-Network architecture.
        
        Args:
            num_actions: Number of possible actions in the action space (default: 2)
        """
        super(DeepQNetwork, self).__init__()
        
        # Convolutional layers for feature extraction from game frames
        # Input: 4 stacked grayscale frames (84x84 each)
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=32,
            kernel_size=8,
            stride=4,
            padding=0
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=0
        )
        
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0
        )
        
        # Fully connected layers for decision making
        # Feature map size after convolutions: 7x7x64 = 3136
        self.fc1 = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)
        
        # Initialize network weights using He initialization for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize network weights using modern initialization techniques.
        
        Uses He initialization for convolutional and linear layers to ensure
        proper gradient flow and faster convergence during training.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # He initialization for convolutional layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # He initialization for linear layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Processes input frames through convolutional layers to extract spatial features,
        then through fully connected layers to produce Q-values for each action.
        
        Args:
            x: Input tensor of shape (batch_size, 4, 84, 84) containing stacked frames
            
        Returns:
            Tensor of shape (batch_size, num_actions) containing Q-values for each action
        """
        # Extract spatial features through convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten feature maps for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Process through fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values
