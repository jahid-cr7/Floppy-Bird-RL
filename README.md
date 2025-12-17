# Deep Q-Learning for Flappy Bird using PyTorch

**Author**: Zonaid Hossain, Abrar Bin Karim, Jahid Hasan

## Project Overview

This project implements a Deep Q-Network (DQN) reinforcement learning agent capable of learning to play the classic Flappy Bird game autonomously. The agent uses convolutional neural networks to process game frames and learn optimal action policies through trial and error, demonstrating the application of deep reinforcement learning in game-playing scenarios.

<p align="center">
  <img src="demo/flappybird.gif" width=600><br/>
  <i>Trained Agent Playing Flappy Bird</i>
</p>

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Reinforcement learning is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. In this project, we apply Deep Q-Learning, a value-based reinforcement learning algorithm, to train an artificial agent to master the Flappy Bird game. The agent learns to navigate through obstacles by maximizing cumulative rewards over time.

### Key Concepts

- **Deep Q-Network (DQN)**: A neural network that approximates the Q-function, which estimates the expected future reward for taking a specific action in a given state.
- **Experience Replay**: A technique that stores past experiences and samples them randomly during training to break correlation between consecutive experiences.
- **Double DQN**: An improvement over standard DQN that uses separate networks for action selection and evaluation to reduce overestimation bias.
- **Epsilon-Greedy Exploration**: A strategy that balances exploration (trying random actions) and exploitation (using learned knowledge) during training.

## Features

- **Modern PyTorch Implementation**: Built using the latest PyTorch features and best practices
- **Double DQN Architecture**: Implements Double DQN for more stable and accurate learning
- **Experience Replay Buffer**: Efficient storage and sampling of training experiences
- **Comprehensive Logging**: TensorBoard integration for monitoring training progress
- **Flexible Configuration**: Extensive command-line arguments for hyperparameter tuning
- **Evaluation Metrics**: Detailed performance statistics during testing
- **Modular Design**: Clean, well-documented code structure for easy understanding and modification

## Architecture

### Neural Network Architecture

The Deep Q-Network consists of:

1. **Convolutional Layers**: Three convolutional layers that extract spatial features from game frames
   - Conv1: 4 input channels → 32 output channels (8×8 kernel, stride 4)
   - Conv2: 32 → 64 channels (4×4 kernel, stride 2)
   - Conv3: 64 → 64 channels (3×3 kernel, stride 1)

2. **Fully Connected Layers**: Two fully connected layers for decision making
   - FC1: 3136 features → 512 hidden units
   - FC2: 512 → 2 output units (Q-values for each action)

3. **Activation Functions**: ReLU activations throughout the network

### Training Algorithm

The training process follows these steps:

1. **State Representation**: Stack 4 consecutive grayscale frames (84×84 pixels each)
2. **Action Selection**: Use epsilon-greedy policy to balance exploration and exploitation
3. **Experience Storage**: Store state transitions in replay buffer
4. **Batch Training**: Sample random batches from replay buffer
5. **Target Network**: Use separate target network for stable Q-value estimation
6. **Loss Calculation**: Compute MSE loss between predicted and target Q-values
7. **Network Update**: Backpropagate gradients and update policy network

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Flappy-bird-deep-Q-learning-pytorch.git
cd Flappy-bird-deep-Q-learning-pytorch
```

### Step 2: Install Dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

Ensure all dependencies are installed correctly:

```bash
python -c "import torch; import pygame; import cv2; print('All dependencies installed successfully!')"
```

## Usage

### Training a New Model

To train a new Deep Q-Network agent from scratch:

```bash
python train.py
```

#### Training Options

The training script supports various command-line arguments for customization:

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
    --save_frequency 100000 \
    --device cuda
```

**Key Hyperparameters:**
- `--num_iterations`: Total number of training iterations (default: 2,000,000)
- `--batch_size`: Number of experiences per training batch (default: 32)
- `--learning_rate`: Learning rate for optimizer (default: 1e-4)
- `--gamma`: Discount factor for future rewards (default: 0.99)
- `--epsilon_start`: Initial exploration rate (default: 1.0)
- `--epsilon_end`: Final exploration rate (default: 0.01)
- `--epsilon_decay`: Exponential decay rate for epsilon (default: 0.995)

### Testing a Trained Model

To evaluate a trained model:

```bash
python test.py --model_path trained_models/flappy_bird_final.pth --num_episodes 10
```

#### Testing Options

```bash
python test.py \
    --model_path trained_models/flappy_bird_final.pth \
    --num_episodes 10 \
    --render \
    --device cuda
```

**Testing Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--num_episodes`: Number of evaluation episodes (default: 10)
- `--render`: Enable visual rendering of the game
- `--device`: Device to use (auto/cuda/cpu)

### Monitoring Training Progress

View training metrics in TensorBoard:

```bash
tensorboard --logdir tensorboard
```

Then open your browser and navigate to `http://localhost:6006`

## Project Structure

```
Flappy-bird-deep-Q-learning-pytorch/
│
├── src/                          # Source code directory
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
│   └── flappy_bird_final.pth
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
└── README.md                    # This file
```

## Technical Details

### State Representation

- **Input**: 4 stacked grayscale frames (84×84 pixels each)
- **Preprocessing**: 
  - Convert RGB to grayscale
  - Resize to 84×84 pixels
  - Apply binary thresholding
  - Normalize pixel values to [0, 1]

### Action Space

- **Action 0**: No flap (bird falls due to gravity)
- **Action 1**: Flap (bird moves upward)

### Reward Structure

- **+1.0**: Successfully passing through a pipe
- **+0.1**: Surviving each frame (small positive reward)
- **-1.0**: Collision with pipe or ground (episode termination)

### Training Improvements

1. **Double DQN**: Reduces overestimation bias by using separate networks for action selection and evaluation
2. **Experience Replay**: Breaks correlation between consecutive experiences
3. **Target Network**: Provides stable Q-value targets during training
4. **Gradient Clipping**: Prevents exploding gradients
5. **He Initialization**: Better weight initialization for faster convergence

## Results

After training for 2 million iterations, the agent typically achieves:

- **Average Score**: 50-100+ points per episode
- **Success Rate**: 90%+ episodes with score > 0
- **Learning Curve**: Steady improvement over training iterations

Training progress can be monitored through TensorBoard, showing:
- Training loss over time
- Q-value estimates
- Episode rewards and scores
- Exploration rate (epsilon) decay

## Contributing

Contributions are welcome! If you'd like to contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Potential Improvements

- Implement Dueling DQN architecture
- Add Prioritized Experience Replay
- Implement Rainbow DQN (combining multiple improvements)
- Add support for different game environments
- Create visualization tools for analyzing agent behavior

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Original Flappy Bird game concept
- DeepMind's DQN paper for the foundational algorithm
- PyTorch community for excellent documentation and tools

## References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.
2. Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.
3. Schaul, T., et al. (2015). "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952.

## Contact

**Author**: Zonaid Hossain, Abrar Bin Karim, Jahid Hasan

For questions, suggestions, or issues, please open an issue on the GitHub repository.

---

**Note**: This project is intended for educational purposes and demonstrates the application of deep reinforcement learning in game-playing scenarios. The implementation follows modern best practices and can serve as a foundation for more advanced reinforcement learning projects.
#   F l o p p y - B i r d - R L  
 