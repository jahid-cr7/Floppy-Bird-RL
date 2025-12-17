"""
Deep Q-Learning Training Script for Flappy Bird
"""
import sys
import os
import traceback

# Redirect stderr to suppress libpng warnings
sys.stderr = open(os.devnull, 'w')

import pygame
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import preprocess_frame


class DQNAgent:
    def __init__(self, state_shape, num_actions, device, learning_rate=0.0001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, memory_size=10000):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Neural networks
        self.policy_net = DeepQNetwork(num_actions).to(device)
        self.target_net = DeepQNetwork(num_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = 32
        
        # Update target network
        self.update_target_net()
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state = state.to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
                
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
            
        try:
            # Sample batch from memory
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Filter out None states
            valid_indices = [i for i, ns in enumerate(next_states) if ns is not None]
            
            if len(valid_indices) == 0:
                return 0
                
            # Convert to tensors
            states = torch.cat([states[i] for i in valid_indices]).to(self.device)
            actions = torch.tensor([actions[i] for i in valid_indices], device=self.device).unsqueeze(1)
            rewards = torch.tensor([rewards[i] for i in valid_indices], device=self.device, dtype=torch.float32)
            next_states = torch.cat([next_states[i] for i in valid_indices]).to(self.device)
            dones = torch.tensor([dones[i] for i in valid_indices], device=self.device, dtype=torch.bool)
            
            # Current Q values
            current_q = self.policy_net(states).gather(1, actions).squeeze()
            
            # Next Q values
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                next_q[dones] = 0.0
                target_q = rewards + (self.gamma * next_q)
            
            # Compute loss
            loss = nn.MSELoss()(current_q, target_q)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            # Decay epsilon
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
                
            return loss.item()
            
        except Exception as e:
            print(f"Error in replay: {e}")
            traceback.print_exc()
            return 0


def train_dqn():
    try:
        # Configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        image_size = 84
        num_actions = 2
        state_shape = (4, image_size, image_size)
        
        # Initialize agent and environment
        agent = DQNAgent(state_shape, num_actions, device)
        game = FlappyBird()
        
        # Training parameters
        episodes = 10000
        target_update = 1000
        print_interval = 100
        
        print("=== STARTING TRAINING ===")
        print(f"Episodes: {episodes}")
        print(f"Initial epsilon: {agent.epsilon}")
        print(f"Memory size: {len(agent.memory)}")
        
        # Training loop
        for episode in range(episodes):
            try:
                # Reset environment
                image, reward, terminal = game.next_frame(0)
                image = preprocess_frame(
                    image[:game.screen_width, :int(game.base_y)],
                    image_size, image_size
                )
                image = torch.from_numpy(image).float().squeeze(0)
                state = torch.stack([image] * 4, dim=0).unsqueeze(0)
                
                total_reward = 0
                steps = 0
                episode_actions = []
                episode_losses = []
                
                while not terminal:
                    # Get action
                    action = agent.act(state)
                    episode_actions.append(action)
                    
                    # Execute action
                    next_image, reward, terminal = game.next_frame(action)
                    next_image = preprocess_frame(
                        next_image[:game.screen_width, :int(game.base_y)],
                        image_size, image_size
                    )
                    next_image = torch.from_numpy(next_image).float().squeeze(0)
                    
                    if not terminal:
                        next_state = torch.cat([state[0, 1:], next_image.unsqueeze(0)], dim=0).unsqueeze(0)
                    else:
                        next_state = None
                        
                    # Store experience
                    agent.remember(state, action, reward, next_state, terminal)
                    
                    # Train agent
                    loss = agent.replay()
                    if loss > 0:
                        episode_losses.append(loss)
                    
                    state = next_state if next_state is not None else state
                    total_reward += reward
                    steps += 1
                    
                    # Safety break to prevent infinite loops
                    if steps > 1000:
                        print("Safety break: too many steps")
                        break

                # Update target network
                if episode % target_update == 0:
                    agent.update_target_net()
                    
                # Print progress
                if episode % print_interval == 0:
                    if episode_actions:
                        actions_count = np.bincount(episode_actions, minlength=num_actions)
                        action_percentages = actions_count / len(episode_actions) * 100
                    else:
                        action_percentages = [0, 0]
                    
                    avg_loss = np.mean(episode_losses) if episode_losses else 0
                    
                    print(f"Episode {episode}")
                    print(f"  Score: {game.score}, Steps: {steps}, Total Reward: {total_reward:.2f}")
                    print(f"  Epsilon: {agent.epsilon:.3f}, Memory: {len(agent.memory)}")
                    print(f"  Actions: {action_percentages[0]:.1f}% no-flap, {action_percentages[1]:.1f}% flap")
                    print(f"  Avg Loss: {avg_loss:.2f}")
                    print("-" * 50)
                    
                # Reset game for next episode
                game = FlappyBird()
                
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                traceback.print_exc()
                # Reset game and continue
                game = FlappyBird()
                continue
    
    except Exception as e:
        print(f"Fatal error in training: {e}")
        traceback.print_exc()
    
    finally:
        # Save trained model
        try:
            os.makedirs("trained_models", exist_ok=True)
            torch.save(agent.policy_net.state_dict(), "trained_models/flappy_bird_trained.pth")
            print("Training completed! Model saved to trained_models/flappy_bird_trained.pth")
        except:
            print("Could not save model")


if __name__ == "__main__":
    train_dqn()