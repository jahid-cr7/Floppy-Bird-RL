"""
Utility Functions for Image Preprocessing

This module provides utility functions for preprocessing game frames
before feeding them into the neural network.

@author: Zonaid Hossain, Abrar Bin Karim, Jahid Hasan
"""
import cv2
import numpy as np
from typing import Union


def preprocess_frame(
    image: np.ndarray,
    width: int,
    height: int,
    normalize: bool = False
) -> np.ndarray:
    """
    Preprocess a game frame for neural network input.
    
    Converts the input image to grayscale, resizes it to the specified dimensions,
    applies binary thresholding, and optionally normalizes pixel values.
    
    Args:
        image: Input RGB image array from the game
        width: Target width for resizing
        height: Target height for resizing
        normalize: Whether to normalize pixel values to [0, 1] range (default: False)
        
    Returns:
        Preprocessed image array with shape (1, height, width) and dtype float32
    """
    # Convert BGR to grayscale (OpenCV uses BGR by default)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image to target dimensions
    resized_image = cv2.resize(grayscale_image, (width, height), interpolation=cv2.INTER_AREA)
    
    # Apply binary thresholding to enhance contrast
    # Pixels with value > 1 become 255, others become 0
    _, binary_image = cv2.threshold(resized_image, 1, 255, cv2.THRESH_BINARY)
    
    # Add channel dimension and convert to float32
    processed_image = binary_image[None, :, :].astype(np.float32)
    
    # Normalize pixel values to [0, 1] range if requested
    if normalize:
        processed_image = processed_image / 255.0
    
    return processed_image


def stack_frames(
    frames: list,
    num_frames: int = 4
) -> np.ndarray:
    """
    Stack multiple frames along the channel dimension.
    
    Creates a state representation by stacking consecutive frames,
    which helps the network understand temporal dynamics and motion.
    
    Args:
        frames: List of preprocessed frame arrays
        num_frames: Number of frames to stack (default: 4)
        
    Returns:
        Stacked frames array with shape (num_frames, height, width)
    """
    if len(frames) < num_frames:
        # Pad with the first frame if not enough frames available
        frames = [frames[0]] * (num_frames - len(frames)) + frames
    
    return np.concatenate(frames[-num_frames:], axis=0)
