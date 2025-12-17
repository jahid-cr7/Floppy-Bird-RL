"""
Deep Q-Learning Model Testing Script for Flappy Bird

This script loads a trained Deep Q-Network model and evaluates its performance
by playing Flappy Bird using the learned policy.

@author: Zonaid Hossain, Abrar Bin Karim, Jahid Hasan
"""
import sys
import os

# Redirect stderr to suppress libpng warnings
sys.stderr = open(os.devnull, 'w')

# Your other imports and code
import pygame
from src.deep_q_network import DeepQNetwork
# ... rest of your code

# Restore stderr if needed later in your program
# sys.stderr = sys.__stderr__
import argparse
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import preprocess_frame


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for testing configuration.
    
    Returns:
        Parsed arguments containing testing parameters
    """
    parser = argparse.ArgumentParser(
        description="Test a trained Deep Q-Network agent playing Flappy Bird"
    )
    
    parser.add_argument(
        "--image_size",
        type=int,
        default=84,
        help="Width and height of preprocessed game frames (default: 84)"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="trained_models/flappy_bird_final.pth",
        help="Path to the trained model checkpoint (default: trained_models/flappy_bird_final.pth)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference (default: auto)"
    )
    
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of episodes to run for evaluation (default: 10)"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether to render the game visually (default: False)"
    )
    
    return parser.parse_args()


def load_model(
    model_path: str,
    device: torch.device
) -> DeepQNetwork:
    """
    Load a trained Deep Q-Network model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint file
        device: Device to load the model onto
        
    Returns:
        Loaded and initialized Deep Q-Network model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize model architecture
    model = DeepQNetwork(num_actions=2).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint (iteration: {checkpoint.get('iteration', 'unknown')})")
        if 'best_score' in checkpoint:
            print(f"Best score during training: {checkpoint['best_score']}")
    else:
        # Legacy format: model saved directly
        model.load_state_dict(checkpoint)
        print("Loaded model from legacy format")
    
    model.eval()
    return model


def evaluate_episode(
    model: DeepQNetwork,
    game: FlappyBird,
    image_size: int,
    device: torch.device,
    render: bool = True
) -> dict:
    """
    Evaluate the model's performance for a single episode.
    
    Args:
        model: Trained Deep Q-Network model
        game: Flappy Bird game environment
        image_size: Size of preprocessed frames
        device: Device to run inference on
        render: Whether to render the game visually
        
    Returns:
        Dictionary containing episode statistics (score, steps, reward)
    """
    # Initialize game state
    image, reward, terminal = game.next_frame(0)
    image = preprocess_frame(
        image[:game.screen_width, :int(game.base_y)],
        image_size,
        image_size
    )
    image = torch.from_numpy(image).float().squeeze(0)  # Remove channel dim: (1, H, W) -> (H, W)
    state = torch.stack([image] * 4, dim=0).unsqueeze(0).to(device)  # (1, 4, H, W)
    
    episode_reward = 0.0
    episode_steps = 0
    
    # Run episode
    while not terminal:
        # Select action using the trained policy (no exploration)
        with torch.no_grad():
            q_values = model(state)
            action = q_values.max(1)[1].item()
        
        # Execute action
        next_image, reward, terminal = game.next_frame(action)
        next_image = preprocess_frame(
            next_image[:game.screen_width, :int(game.base_y)],
            image_size,
            image_size
        )
        next_image = torch.from_numpy(next_image).float().squeeze(0)  # Remove channel dim: (1, H, W) -> (H, W)
        next_state = torch.cat([state[0, 1:], next_image.unsqueeze(0)], dim=0).unsqueeze(0).to(device)  # (1, 4, H, W)
        
        # Update statistics
        episode_reward += reward
        episode_steps += 1
        state = next_state
        
        # Small delay for visualization
        if render:
            time.sleep(0.01)
    
    return {
        'score': game.score,
        'steps': episode_steps,
        'reward': episode_reward
    }


def test(config: argparse.Namespace) -> None:
    """
    Main testing function for evaluating trained model performance.
    
    Args:
        config: Testing configuration parameters
    """
    # Set up device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    # Load trained model
    print(f"Loading model from: {config.model_path}")
    model = load_model(config.model_path, device)
    
    # Initialize game environment
    game = FlappyBird()
    
    # Evaluation statistics
    scores = []
    steps = []
    rewards = []
    
    print(f"\nRunning {config.num_episodes} evaluation episodes...")
    print("-" * 60)
    
    # Run evaluation episodes
    for episode in range(config.num_episodes):
        episode_stats = evaluate_episode(
            model, game, config.image_size, device, config.render
        )
        
        scores.append(episode_stats['score'])
        steps.append(episode_stats['steps'])
        rewards.append(episode_stats['reward'])
        
        print(f"Episode {episode + 1}/{config.num_episodes}: "
              f"Score = {episode_stats['score']}, "
              f"Steps = {episode_stats['steps']}, "
              f"Reward = {episode_stats['reward']:.2f}")
    
    # Print summary statistics
    print("-" * 60)
    print("\nEvaluation Summary:")
    print(f"  Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"  Best Score: {np.max(scores)}")
    print(f"  Worst Score: {np.min(scores)}")
    print(f"  Average Steps: {np.mean(steps):.2f} ± {np.std(steps):.2f}")
    print(f"  Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Success Rate (Score > 0): {sum(s > 0 for s in scores) / len(scores) * 100:.1f}%")


if __name__ == "__main__":
    args = parse_arguments()
    test(args)
