# Deep Q-Learning for Flappy Bird - Complete Documentation

**Author**: Zonaid Hossain, Abrar Bin Karim, Jahid Hasan

## Project Overview

This project implements a Deep Q-Network (DQN) reinforcement learning agent that learns to play Flappy Bird autonomously. The implementation uses PyTorch and follows modern deep reinforcement learning best practices, including Double DQN, experience replay, and target networks.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage Guide](#usage-guide)
5. [Training Results](#training-results)
6. [Testing Results](#testing-results)
7. [Technical Architecture](#technical-architecture)
8. [Hyperparameters](#hyperparameters)
9. [Troubleshooting](#troubleshooting)
10. [Future Improvements](#future-improvements)

---

## System Requirements

### Hardware
- **CPU**: Any modern processor (Intel/AMD)
- **GPU**: Optional but recommended for faster training (CUDA-compatible)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: ~500MB for project files and models

### Software
- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.8 or higher
- **CUDA**: Optional, version 11.0+ if using GPU acceleration

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Flappy-bird-deep-Q-learning-pytorch
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; import pygame; import cv2; print('All dependencies installed successfully!')"
```

---

## Project Structure

```
Flappy-bird-deep-Q-learning-pytorch/
│
├── src/                          # Source code directory
│   ├── __init__.py              # Package initialization
│   ├── deep_q_network.py        # DQN architecture implementation
│   ├── flappy_bird.py           # Game environment implementation
│   └── utils.py                 # Utility functions for preprocessing
│
├── assets/                       # Game assets
│   └── sprites/                 # Game sprite images
│       ├── background-black.png
│       ├── base.png
│       ├── pipe-green.png
│       └── redbird-*.png
│
├── trained_models/              # Saved model checkpoints
│   ├── flappy_bird_final.pth
│   ├── flappy_bird_10000.pth
│   └── flappy_bird_20000.pth
│
├── tensorboard/                 # TensorBoard log files
│
├── demo/                        # Demo videos and GIFs
│   ├── flappybird.gif
│   └── flappybird.mp4
│
├── train.py                     # Training script
├── test.py                      # Testing/evaluation script
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
└── DOCUMENTATION.md             # This file
```

---

## Usage Guide

### Training a Model

#### Basic Training

Train with default hyperparameters:

```bash
python train.py
```

#### Custom Training Configuration

```bash
python train.py \
    --num_iterations 2000000 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --gamma 0.99 \
    --epsilon_start 1.0 \
    --epsilon_end 0.01 \
    --epsilon_decay 0.995 \
    --replay_memory_size 50000 \
    --update_target_frequency 10000 \
    --start_training 10000 \
    --save_frequency 100000 \
    --device cuda
```

#### Training Parameters Explained

- `--num_iterations`: Total number of training iterations (default: 2,000,000)
- `--batch_size`: Number of experiences per training batch (default: 32)
- `--learning_rate`: Learning rate for optimizer (default: 1e-4)
- `--gamma`: Discount factor for future rewards (default: 0.99)
- `--epsilon_start`: Initial exploration rate (default: 1.0)
- `--epsilon_end`: Final exploration rate (default: 0.01)
- `--epsilon_decay`: Exponential decay rate for epsilon (default: 0.995)
- `--replay_memory_size`: Maximum size of experience replay buffer (default: 50,000)
- `--update_target_frequency`: Frequency of target network updates (default: 10,000)
- `--start_training`: Number of random steps before training starts (default: 10,000)
- `--save_frequency`: Frequency of model checkpoint saves (default: 100,000)
- `--device`: Device to use (auto/cuda/cpu, default: auto)

### Testing a Model

#### Basic Testing

```bash
python test.py --model_path trained_models/flappy_bird_final.pth
```

#### Custom Testing Configuration

```bash
python test.py \
    --model_path trained_models/flappy_bird_final.pth \
    --num_episodes 10 \
    --render \
    --device cuda
```

#### Testing Parameters Explained

- `--model_path`: Path to trained model checkpoint
- `--num_episodes`: Number of evaluation episodes (default: 10)
- `--render`: Enable visual rendering of the game (default: False)
- `--device`: Device to use (auto/cuda/cpu, default: auto)

### Monitoring Training Progress

View training metrics in TensorBoard:

```bash
tensorboard --logdir tensorboard
```

Then open your browser and navigate to `http://localhost:6006`

---

## Training Results

### Training Session Summary

**Configuration Used:**
- Total Iterations: 50,000 (test run)
- Batch Size: 32
- Learning Rate: 1e-4
- Epsilon Decay: Exponential (0.995)
- Device: CPU

**Training Metrics:**
- Training Speed: ~27-29 iterations/second
- Loss Values: Decreasing from ~100,000 to ~3,000-5,000 range
- Epsilon Decay: From 1.0 to 0.01 (exploration to exploitation)
- Model Checkpoints: Saved at iterations 10,000, 20,000, and final

**Observations:**
- Loss decreased steadily throughout training
- Network learned to reduce Q-value estimation errors
- Epsilon decayed smoothly, transitioning from exploration to exploitation
- Training was stable with no crashes or errors

### Model Checkpoints

The following model checkpoints were created during training:

1. **flappy_bird_10000.pth**: Model at 10,000 iterations
2. **flappy_bird_20000.pth**: Model at 20,000 iterations
3. **flappy_bird_final.pth**: Final model at training completion

---

## Testing Results

### Test Configuration

- **Model Tested**: `flappy_bird_20000.pth` (20,000 iterations)
- **Number of Episodes**: 5
- **Device**: CPU
- **Rendering**: Disabled (for faster testing)

### Test Results Summary

```
Evaluation Summary:
  Average Score: 0.00 ± 0.00
  Best Score: 0
  Worst Score: 0
  Average Steps: 49.00 ± 0.00
  Average Reward: 3.80 ± 0.00
  Success Rate (Score > 0): 0.0%
```

### Analysis

**Performance Metrics:**
- **Average Score**: 0 (agent did not pass any pipes)
- **Average Steps**: 49 steps per episode
- **Average Reward**: 3.80 (survival reward)
- **Success Rate**: 0% (no successful pipe passages)

**Interpretation:**
- The model at 20,000 iterations is still in early learning phase
- Agent consistently survives ~49 steps before collision
- No successful pipe passages observed, indicating need for more training
- Model is learning basic survival but not yet optimal navigation

**Expected Improvement with More Training:**
- With full 2,000,000 iterations, the agent should achieve:
  - Average scores of 50-100+ points
  - Success rate of 90%+ episodes with score > 0
  - Better navigation through pipe gaps

---

## Technical Architecture

### Deep Q-Network Architecture

The neural network consists of:

1. **Convolutional Layers**:
   - Conv1: 4 channels → 32 channels (8×8 kernel, stride 4)
   - Conv2: 32 → 64 channels (4×4 kernel, stride 2)
   - Conv3: 64 → 64 channels (3×3 kernel, stride 1)

2. **Fully Connected Layers**:
   - FC1: 3136 features → 512 hidden units
   - FC2: 512 → 2 output units (Q-values for each action)

3. **Activation Functions**: ReLU throughout

4. **Weight Initialization**: He initialization for better convergence

### Training Algorithm

The training process implements:

1. **Double DQN**: Uses separate networks for action selection and evaluation
2. **Experience Replay**: Stores and samples past experiences randomly
3. **Target Network**: Provides stable Q-value targets during training
4. **Epsilon-Greedy Exploration**: Balances exploration and exploitation
5. **Gradient Clipping**: Prevents exploding gradients

### State Representation

- **Input**: 4 stacked grayscale frames (84×84 pixels each)
- **Preprocessing**:
  - Convert RGB to grayscale
  - Resize to 84×84 pixels
  - Apply binary thresholding
  - Stack 4 consecutive frames

### Action Space

- **Action 0**: No flap (bird falls due to gravity)
- **Action 1**: Flap (bird moves upward)

### Reward Structure

- **+1.0**: Successfully passing through a pipe
- **+0.1**: Surviving each frame (small positive reward)
- **-1.0**: Collision with pipe or ground (episode termination)

---

## Hyperparameters

### Recommended Hyperparameters

For optimal training performance:

```python
{
    'num_iterations': 2000000,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'replay_memory_size': 50000,
    'update_target_frequency': 10000,
    'start_training': 10000,
    'save_frequency': 100000
}
```

### Hyperparameter Tuning Tips

1. **Learning Rate**: Start with 1e-4, adjust if loss doesn't decrease
2. **Batch Size**: Larger batches (64) may improve stability but slower training
3. **Epsilon Decay**: Slower decay (0.99) allows more exploration
4. **Gamma**: Higher values (0.99) emphasize long-term rewards
5. **Replay Memory Size**: Larger buffers (100,000) store more diverse experiences

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: ModuleNotFoundError

**Problem**: Missing dependencies

**Solution**:
```bash
pip install -r requirements.txt
```

#### Issue 2: CUDA Out of Memory

**Problem**: GPU memory insufficient

**Solution**:
- Reduce batch size: `--batch_size 16`
- Use CPU: `--device cpu`
- Reduce replay memory size

#### Issue 3: Training Loss Not Decreasing

**Problem**: Learning rate too high or network not learning

**Solution**:
- Lower learning rate: `--learning_rate 1e-5`
- Increase training iterations
- Check if epsilon is decaying properly

#### Issue 4: Model Not Performing Well

**Problem**: Insufficient training

**Solution**:
- Train for more iterations (2M+ recommended)
- Ensure epsilon decays to low value (0.01)
- Check TensorBoard metrics for learning progress

#### Issue 5: Game Window Not Displaying

**Problem**: Display issues on headless systems

**Solution**:
- Use `--render` flag only when display available
- Test without rendering for faster evaluation

---

## Future Improvements

### Potential Enhancements

1. **Dueling DQN**: Separate value and advantage streams
2. **Prioritized Experience Replay**: Sample important experiences more frequently
3. **Rainbow DQN**: Combine multiple DQN improvements
4. **Noisy Networks**: Replace epsilon-greedy with learned exploration
5. **Distributional RL**: Model full reward distribution instead of mean

### Code Improvements

1. Add support for multiple game environments
2. Implement model checkpoint resuming
3. Add hyperparameter optimization tools
4. Create visualization tools for agent behavior
5. Add unit tests for all components

---

## References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.

2. Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.

3. Schaul, T., et al. (2015). "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952.

---

## License

This project is open source and available under the MIT License.

---

## Contact and Support

**Author**: Zonaid Hossain, Abrar Bin Karim, Jahid Hasan

For questions, issues, or contributions:
- Open an issue on the GitHub repository
- Check the README.md for basic usage
- Review TensorBoard logs for training diagnostics

---

**Last Updated**: 2024
**Version**: 2.0.0

