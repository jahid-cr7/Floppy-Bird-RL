"""
Flappy Bird Game Environment Implementation

This module implements the Flappy Bird game environment for reinforcement learning.
It provides a game interface compatible with OpenAI Gym-style environments,
allowing the agent to interact with the game through actions and receive rewards.

@author: Zonaid Hossain, Abrar Bin Karim, Jahid Hasan


"""
import os
# Suppress pygame welcome message and other warnings
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
import warnings
warnings.filterwarnings("ignore")

import pygame
from itertools import cycle
from typing import Tuple
import numpy as np
from numpy.random import randint
import pygame
from pygame import Rect, init, time, display
from pygame.event import pump
from pygame.image import load
from pygame.surfarray import array3d, pixels_alpha
from pygame.transform import rotate


class FlappyBird:
    """
    Flappy Bird game environment for reinforcement learning.
    
    This class implements a complete Flappy Bird game environment where an agent
    can learn to play by taking actions (flap or no-flap) and receiving rewards
    based on its performance. The environment provides visual feedback and
    collision detection for training deep reinforcement learning agents.
    
    Attributes:
        screen_width: Width of the game screen in pixels
        screen_height: Height of the game screen in pixels
        fps: Frames per second for game rendering
        pipe_gap_size: Vertical gap between upper and lower pipes
        pipe_velocity_x: Horizontal velocity of moving pipes
    """
    
    # Class-level constants (shared across all instances)
    init()
    fps_clock = time.Clock()
    screen_width = 288
    screen_height = 512
    screen = display.set_mode((screen_width, screen_height))
    display.set_caption('Deep Q-Network Flappy Bird')
    
    # Load game assets
    base_image = load('assets/sprites/base.png').convert_alpha()
    background_image = load('assets/sprites/background-black.png').convert()
    
    pipe_images = [
        rotate(load('assets/sprites/pipe-green.png').convert_alpha(), 180),
        load('assets/sprites/pipe-green.png').convert_alpha()
    ]
    
    bird_images = [
        load('assets/sprites/redbird-upflap.png').convert_alpha(),
        load('assets/sprites/redbird-midflap.png').convert_alpha(),
        load('assets/sprites/redbird-downflap.png').convert_alpha()
    ]
    
    # Create collision masks for pixel-perfect collision detection
    bird_hitmask = [pixels_alpha(image).astype(bool) for image in bird_images]
    pipe_hitmask = [pixels_alpha(image).astype(bool) for image in pipe_images]
    
    # Game physics parameters
    fps = 30
    pipe_gap_size = 100
    pipe_velocity_x = -4
    
    # Bird physics parameters
    min_velocity_y = -8
    max_velocity_y = 10
    downward_speed = 1
    upward_speed = -9
    
    # Bird animation cycle
    bird_index_generator = cycle([0, 1, 2, 1])
    
    def __init__(self) -> None:
        """
        Initialize a new Flappy Bird game instance.
        
        Sets up the game state, initializes the bird position, creates initial
        pipes, and resets all game variables to their starting values.
        """
        # Game state variables
        self.iter = 0
        self.bird_index = 0
        self.score = 0
        
        # Get sprite dimensions
        self.bird_width = self.bird_images[0].get_width()
        self.bird_height = self.bird_images[0].get_height()
        self.pipe_width = self.pipe_images[0].get_width()
        self.pipe_height = self.pipe_images[0].get_height()
        
        # Initialize bird position (left side, middle of screen)
        self.bird_x = int(self.screen_width / 5)
        self.bird_y = int((self.screen_height - self.bird_height) / 2)
        
        # Initialize base (ground) position
        self.base_x = 0
        self.base_y = self.screen_height * 0.79
        self.base_shift = self.base_image.get_width() - self.background_image.get_width()
        
        # Create initial pipes
        pipes = [self.generate_pipe(), self.generate_pipe()]
        pipes[0]["x_upper"] = pipes[0]["x_lower"] = self.screen_width
        pipes[1]["x_upper"] = pipes[1]["x_lower"] = self.screen_width * 1.5
        self.pipes = pipes
        
        # Bird physics state
        self.current_velocity_y = 0
        self.is_flapped = False
    
    def generate_pipe(self) -> dict:
        """
        Generate a new pipe obstacle with random vertical position.
        
        Creates a pipe pair (upper and lower) with a gap between them.
        The gap position is randomly generated within a valid range.
        
        Returns:
            Dictionary containing pipe positions:
            - x_upper: X position of upper pipe
            - y_upper: Y position of upper pipe
            - x_lower: X position of lower pipe
            - y_lower: Y position of lower pipe
        """
        x = self.screen_width + 10
        # Generate gap position randomly within valid range
        gap_y = randint(2, 10) * 10 + int(self.base_y / 5)
        
        return {
            "x_upper": x,
            "y_upper": gap_y - self.pipe_height,
            "x_lower": x,
            "y_lower": gap_y + self.pipe_gap_size
        }
    
    def is_collided(self) -> bool:
        """
        Check if the bird has collided with pipes or ground.
        
        Performs pixel-perfect collision detection between the bird sprite
        and pipe sprites, as well as checking if the bird has hit the ground.
        
        Returns:
            True if collision detected, False otherwise
        """
        # Check ground collision
        if self.bird_height + self.bird_y + 1 >= self.base_y:
            return True
        
        # Create bounding box for bird
        bird_bbox = Rect(self.bird_x, self.bird_y, self.bird_width, self.bird_height)
        pipe_boxes = []
        
        # Create bounding boxes for all pipes
        for pipe in self.pipes:
            pipe_boxes.append(Rect(pipe["x_upper"], pipe["y_upper"], 
                                 self.pipe_width, self.pipe_height))
            pipe_boxes.append(Rect(pipe["x_lower"], pipe["y_lower"], 
                                 self.pipe_width, self.pipe_height))
        
        # Check bounding box overlap
        if bird_bbox.collidelist(pipe_boxes) == -1:
            return False
        
        # Perform pixel-perfect collision detection
        for i, pipe_box in enumerate(pipe_boxes):
            cropped_bbox = bird_bbox.clip(pipe_box)
            
            if cropped_bbox.width == 0 or cropped_bbox.height == 0:
                continue
            
            # Calculate relative positions for mask comparison
            min_x1 = cropped_bbox.x - bird_bbox.x
            min_y1 = cropped_bbox.y - bird_bbox.y
            min_x2 = cropped_bbox.x - pipe_box.x
            min_y2 = cropped_bbox.y - pipe_box.y
            
            # Check pixel-level collision using masks
            pipe_mask_index = i % 2
            if np.any(
                self.bird_hitmask[self.bird_index][
                    min_x1:min_x1 + cropped_bbox.width,
                    min_y1:min_y1 + cropped_bbox.height
                ] * self.pipe_hitmask[pipe_mask_index][
                    min_x2:min_x2 + cropped_bbox.width,
                    min_y2:min_y2 + cropped_bbox.height
                ]
            ):
                return True
        
        return False
    
    def next_frame(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one game step based on the given action.
        
        Updates the game state, processes the action, checks for collisions,
        updates the score, and renders the game frame.
        
        Args:
            action: Action to take (0 = no flap, 1 = flap)
            
        Returns:
            Tuple containing:
            - image: Current game frame as numpy array (RGB)
            - reward: Reward received for this step
            - terminal: Whether the episode has ended
        """
        # Process pygame events
        pump()
        
        # Initialize reward and terminal flag
        reward = 0.1  # Small positive reward for survival
        terminal = False
        
        # Process action: flap if action is 1
        if action == 1:
            self.current_velocity_y = self.upward_speed
            self.is_flapped = True
        
        # Update score when bird passes a pipe
        bird_center_x = self.bird_x + self.bird_width / 2
        for pipe in self.pipes:
            pipe_center_x = pipe["x_upper"] + self.pipe_width / 2
            # Check if bird has passed the pipe
            if pipe_center_x < bird_center_x < pipe_center_x + 5:
                self.score += 1
                reward = 1.0  # Large positive reward for passing a pipe
                break
        
        # Update bird animation
        if (self.iter + 1) % 3 == 0:
            self.bird_index = next(self.bird_index_generator)
            self.iter = 0
        self.iter += 1
        
        # Update base (ground) scrolling
        self.base_x = -((-self.base_x + 100) % self.base_shift)
        
        # Update bird physics
        if self.current_velocity_y < self.max_velocity_y and not self.is_flapped:
            self.current_velocity_y += self.downward_speed
        
        if self.is_flapped:
            self.is_flapped = False
        
        # Update bird position
        self.bird_y += min(
            self.current_velocity_y,
            self.bird_y - self.current_velocity_y - self.bird_height
        )
        
        # Prevent bird from going above screen
        if self.bird_y < 0:
            self.bird_y = 0
        
        # Update pipe positions
        for pipe in self.pipes:
            pipe["x_upper"] += self.pipe_velocity_x
            pipe["x_lower"] += self.pipe_velocity_x
        
        # Generate new pipes and remove off-screen pipes
        if 0 < self.pipes[0]["x_lower"] < 5:
            self.pipes.append(self.generate_pipe())
        
        if self.pipes[0]["x_lower"] < -self.pipe_width:
            del self.pipes[0]
        
        # Check for collision
        if self.is_collided():
            terminal = True
            reward = -1.0  # Negative reward for collision
            # Reset game state
            self.__init__()
        
        # Render game frame
        self.screen.blit(self.background_image, (0, 0))
        self.screen.blit(self.base_image, (self.base_x, self.base_y))
        self.screen.blit(self.bird_images[self.bird_index], (self.bird_x, self.bird_y))
        
        for pipe in self.pipes:
            self.screen.blit(self.pipe_images[0], (pipe["x_upper"], pipe["y_upper"]))
            self.screen.blit(self.pipe_images[1], (pipe["x_lower"], pipe["y_lower"]))
        
        # Convert screen to numpy array
        image = array3d(display.get_surface())
        display.update()
        self.fps_clock.tick(self.fps)
        
        return image, reward, terminal
