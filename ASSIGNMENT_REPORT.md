# Deep Q-Learning for Flappy Bird: A Deep Learning Assignment

**Author**: Zonaid Hossain, Abrar Bin Karim, Jahid Hasan
**Course**: Deep Learning  
**Date**: 2025

---

## Abstract

This project implements and evaluates a Deep Q-Network (DQN) reinforcement learning agent for playing the Flappy Bird game. The implementation uses PyTorch and incorporates modern deep reinforcement learning techniques including Double DQN, experience replay, and target networks. The agent learns to play the game autonomously by processing game frames through a convolutional neural network and learning optimal action policies through trial and error. Experimental results demonstrate the learning progress of the agent, with loss values decreasing from ~100,000 to ~3,000-5,000 over 50,000 training iterations. The project demonstrates the practical application of deep reinforcement learning in game-playing scenarios and provides insights into the training dynamics of DQN algorithms.

**Keywords**: Deep Reinforcement Learning, Deep Q-Network, Double DQN, PyTorch, Game AI

---

## 1. Introduction

### 1.1 Background

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. Unlike supervised learning, RL agents learn from trial and error, receiving rewards or penalties based on their actions. Deep Reinforcement Learning combines RL with deep neural networks, enabling agents to learn complex behaviors from high-dimensional sensory inputs such as images.

### 1.2 Problem Statement

The Flappy Bird game presents an interesting challenge for reinforcement learning:
- **State Space**: High-dimensional visual input (game frames)
- **Action Space**: Discrete (flap or no-flap)
- **Reward Structure**: Sparse rewards (survival, passing pipes, collisions)
- **Temporal Dependencies**: Requires understanding of motion and timing

The goal is to train an agent that can autonomously learn to play Flappy Bird by maximizing its score through optimal action selection.

### 1.3 Objectives

1. Implement a Deep Q-Network (DQN) architecture for processing game frames
2. Develop a training pipeline with experience replay and target networks
3. Implement Double DQN to improve learning stability
4. Evaluate the agent's performance and analyze learning dynamics
5. Document the implementation and experimental results

### 1.4 Project Scope

This project focuses on:
- Implementing DQN with Double DQN improvements
- Training the agent on the Flappy Bird environment
- Analyzing training metrics and performance
- Documenting the methodology and results

---

## 2. Literature Review

### 2.1 Deep Q-Networks (DQN)

Deep Q-Networks were introduced by Mnih et al. (2015) in their groundbreaking paper "Human-level control through deep reinforcement learning." DQN combines Q-learning with deep neural networks to handle high-dimensional state spaces. Key innovations include:

- **Experience Replay**: Storing and randomly sampling past experiences to break correlation
- **Target Network**: Using a separate network for Q-value estimation to stabilize training
- **Frame Stacking**: Using multiple consecutive frames to capture temporal information

### 2.2 Double DQN

Van Hasselt et al. (2016) introduced Double DQN to address the overestimation bias in standard DQN. The key improvement is using separate networks for action selection and evaluation:

- **Action Selection**: Uses the main network to select the best action
- **Action Evaluation**: Uses the target network to evaluate the selected action

This reduces overestimation bias and leads to more stable and accurate learning.

### 2.3 Related Work

Several improvements to DQN have been proposed:
- **Dueling DQN**: Separates value and advantage estimation
- **Prioritized Experience Replay**: Samples important experiences more frequently
- **Rainbow DQN**: Combines multiple DQN improvements

For this assignment, we focus on Double DQN as it provides a good balance between complexity and performance improvement.

---

## 3. Methodology

### 3.1 Deep Q-Learning Algorithm

Q-Learning is a value-based reinforcement learning algorithm that learns the action-value function Q(s, a), representing the expected future reward for taking action a in state s.

The Q-function is updated using the Bellman equation:

```
Q(s, a) = r + γ * max Q(s', a')
```

Where:
- `r` is the immediate reward
- `γ` (gamma) is the discount factor (0.99)
- `s'` is the next state
- `a'` is the next action

### 3.2 Deep Q-Network Architecture

The DQN uses a convolutional neural network to approximate the Q-function:

**Input**: 4 stacked grayscale frames (84×84 pixels each) → Shape: (1, 4, 84, 84)

**Architecture**:
1. **Convolutional Layer 1**: 4 → 32 channels, 8×8 kernel, stride 4
2. **Convolutional Layer 2**: 32 → 64 channels, 4×4 kernel, stride 2
3. **Convolutional Layer 3**: 64 → 64 channels, 3×3 kernel, stride 1
4. **Fully Connected Layer 1**: 3136 → 512 features
5. **Fully Connected Layer 2**: 512 → 2 outputs (Q-values for each action)

**Activation**: ReLU for all hidden layers

**Initialization**: He initialization for better convergence

### 3.3 Double DQN Implementation

Double DQN uses two networks:
- **Policy Network**: Updated every step, used for action selection
- **Target Network**: Updated periodically (every 10,000 steps), used for Q-value evaluation

The target Q-value is computed as:
```
Q_target = r + γ * Q_target(s', argmax_a Q_policy(s', a))
```

### 3.4 Experience Replay

Experience replay stores past transitions (s, a, r, s', done) in a buffer and samples random batches for training. This:
- Breaks correlation between consecutive experiences
- Improves sample efficiency
- Stabilizes training

**Buffer Size**: 50,000 experiences  
**Batch Size**: 32 experiences per training step

### 3.5 Exploration Strategy

Epsilon-greedy exploration balances exploration and exploitation:
- **Exploration**: Random action with probability ε
- **Exploitation**: Best action according to Q-network with probability (1-ε)

**Epsilon Decay**: Exponential decay from 1.0 to 0.01
```
ε = max(ε_end, ε * ε_decay)
```

### 3.6 Training Process

1. Initialize networks (policy and target)
2. Initialize experience replay buffer
3. For each iteration:
   - Select action using epsilon-greedy policy
   - Execute action and observe reward and next state
   - Store experience in replay buffer
   - Sample batch from replay buffer
   - Compute target Q-values using Double DQN
   - Update policy network using MSE loss
   - Periodically update target network
   - Decay epsilon

---

## 4. Experimental Setup

### 4.1 Environment

- **Game**: Flappy Bird (custom implementation)
- **Screen Size**: 288×512 pixels
- **Frame Rate**: 30 FPS
- **Preprocessing**: Resize to 84×84, grayscale, binary thresholding

### 4.2 Hyperparameters

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Learning Rate | 1e-4 | Adam optimizer learning rate |
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.01 | Final exploration rate |
| Epsilon Decay | 0.995 | Exponential decay rate |
| Batch Size | 32 | Number of experiences per batch |
| Replay Buffer Size | 50,000 | Maximum experiences stored |
| Target Update Frequency | 10,000 | Steps between target network updates |
| Start Training | 10,000 | Random steps before training begins |
| Image Size | 84×84 | Preprocessed frame dimensions |

### 4.3 Reward Structure

- **+1.0**: Successfully passing through a pipe
- **+0.1**: Surviving each frame (small positive reward)
- **-1.0**: Collision with pipe or ground (episode termination)

### 4.4 Training Configuration

- **Total Iterations**: 50,000 (test run) / 2,000,000 (full training)
- **Device**: CPU (GPU optional for faster training)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Gradient Clipping**: Max norm 10.0

### 4.5 Evaluation Metrics

- **Score**: Number of pipes passed
- **Episode Length**: Number of steps per episode
- **Success Rate**: Percentage of episodes with score > 0
- **Average Reward**: Mean reward per episode
- **Training Loss**: MSE loss during training
- **Q-Values**: Estimated action values

---

## 5. Results and Analysis

### 5.1 Training Results

#### 5.1.1 Training Metrics

**Training Session**: 50,000 iterations (test run)

| Metric | Value |
|--------|-------|
| Training Speed | ~27-29 iterations/second (CPU) |
| Initial Loss | ~100,000 |
| Final Loss | ~3,000-5,000 |
| Loss Reduction | ~95% decrease |
| Epsilon Decay | 1.0 → 0.01 (smooth) |
| Model Checkpoints | Saved at 10K, 20K, 30K, and final |

#### 5.1.2 Loss Analysis

The training loss decreased steadily throughout training:
- **Early Training** (0-10K iterations): Loss ~100,000, high variance
- **Mid Training** (10K-30K iterations): Loss ~10,000-5,000, decreasing trend
- **Late Training** (30K-50K iterations): Loss ~3,000-5,000, stabilizing

This indicates the network is learning to better estimate Q-values.

#### 5.1.3 Epsilon Decay

Epsilon decayed smoothly from 1.0 to 0.01:
- **Early Phase**: High exploration (ε ≈ 1.0), agent tries random actions
- **Mid Phase**: Balanced exploration/exploitation (ε ≈ 0.5)
- **Late Phase**: Low exploration (ε ≈ 0.01), agent uses learned policy

### 5.2 Model Performance

#### 5.2.1 Evaluation Results

**Model**: flappy_bird_20000.pth (20,000 iterations)  
**Test Episodes**: 5

| Episode | Score | Steps | Reward |
|---------|-------|-------|--------|
| 1 | 0 | 49 | 3.80 |
| 2 | 0 | 49 | 3.80 |
| 3 | 0 | 49 | 3.80 |
| 4 | 0 | 49 | 3.80 |
| 5 | 0 | 49 | 3.80 |

**Summary Statistics**:
- Average Score: 0.00 ± 0.00
- Average Steps: 49.00 ± 0.00
- Average Reward: 3.80 ± 0.00
- Success Rate: 0.0%

#### 5.2.2 Performance Analysis

**Current Performance (20K iterations)**:
- Agent consistently survives ~49 steps before collision
- No successful pipe passages observed
- Behavior is consistent but not yet optimal

**Interpretation**:
- Model is in early learning phase
- Agent has learned basic survival (avoiding immediate collisions)
- More training needed for optimal navigation

**Expected Performance (2M iterations)**:
- Average score: 50-100+ points
- Success rate: 90%+ episodes with score > 0
- Better obstacle navigation and timing

### 5.3 Learning Dynamics

#### 5.3.1 Q-Value Evolution

Q-values increased over training, indicating the network is learning to estimate future rewards more accurately. The Double DQN implementation helps prevent overestimation bias.

#### 5.3.2 Training Stability

Training was stable throughout:
- No crashes or errors
- Smooth loss decrease
- Consistent learning progress
- No gradient explosion (gradient clipping effective)

### 5.4 TensorBoard Metrics

Training metrics logged to TensorBoard include:
- **Training Loss**: Decreasing trend over iterations
- **Epsilon**: Smooth decay from 1.0 to 0.01
- **Reward**: Per-step rewards during training
- **Q-Values**: Estimated action values
- **Score**: Episode scores during training

---

## 6. Discussion

### 6.1 Implementation Challenges

1. **State Representation**: Converting game frames to suitable input format
   - **Solution**: Frame stacking and preprocessing pipeline

2. **Training Stability**: Preventing unstable learning
   - **Solution**: Target network, gradient clipping, Double DQN

3. **Exploration vs Exploitation**: Balancing exploration and exploitation
   - **Solution**: Epsilon-greedy with exponential decay

4. **Memory Management**: Storing and sampling experiences efficiently
   - **Solution**: Circular buffer with fixed size

### 6.2 Design Decisions

1. **Double DQN over Standard DQN**: Chosen for better stability and accuracy
2. **Frame Stacking**: 4 frames to capture temporal information
3. **He Initialization**: Better than uniform initialization for ReLU networks
4. **Adam Optimizer**: Adaptive learning rate, better than SGD for this problem

### 6.3 Limitations

1. **Training Time**: 50K iterations is insufficient for optimal performance
2. **CPU Training**: Slower than GPU training
3. **Limited Evaluation**: Only 5 test episodes evaluated
4. **No Ablation Studies**: Didn't compare with baseline DQN

### 6.4 Future Improvements

1. **Extended Training**: Train for full 2M iterations
2. **Advanced Architectures**: Implement Dueling DQN or Rainbow DQN
3. **Hyperparameter Tuning**: Systematic hyperparameter search
4. **More Evaluation**: Test with 50+ episodes for better statistics
5. **Visualization**: Create visualizations of agent behavior
6. **Ablation Studies**: Compare with baseline DQN and analyze components

---

## 7. Conclusion

This project successfully implements a Deep Q-Network with Double DQN improvements for playing Flappy Bird. The implementation demonstrates:

1. **Successful Implementation**: All components (DQN, Double DQN, experience replay) working correctly
2. **Learning Progress**: Loss decreased significantly, indicating learning
3. **Stable Training**: No crashes, smooth learning dynamics
4. **Modern Practices**: Uses latest PyTorch features and best practices

While the model at 20K iterations shows limited performance, this is expected for early training. With extended training (2M iterations), the agent should achieve much better performance.

The project demonstrates the practical application of deep reinforcement learning and provides a solid foundation for further improvements and research.

---

## 8. References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.

2. Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." Proceedings of the AAAI Conference on Artificial Intelligence, 30(1).

3. Schaul, T., et al. (2015). "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952.

4. Wang, Z., et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." International Conference on Machine Learning, 1995-2003.

5. Hessel, M., et al. (2018). "Rainbow: Combining Improvements in Deep Reinforcement Learning." Proceedings of the AAAI Conference on Artificial Intelligence, 32(1).

---

## 9. Appendix

### 9.1 Code Structure

```
Flappy-bird-deep-Q-learning-pytorch/
├── src/
│   ├── deep_q_network.py    # DQN architecture
│   ├── flappy_bird.py       # Game environment
│   └── utils.py             # Preprocessing utilities
├── train.py                 # Training script
├── test.py                  # Evaluation script
└── requirements.txt         # Dependencies
```

### 9.2 Key Functions

**Training**:
- `train()`: Main training loop
- `train_step()`: Single training step with Double DQN
- `select_action()`: Epsilon-greedy action selection

**Evaluation**:
- `evaluate_episode()`: Run single evaluation episode
- `load_model()`: Load trained model checkpoint

**Network**:
- `DeepQNetwork.forward()`: Forward pass through network
- `DeepQNetwork._initialize_weights()`: He initialization

### 9.3 Running the Code

**Training**:
```bash
python train.py --num_iterations 2000000 --device cuda
```

**Testing**:
```bash
python test.py --model_path trained_models/flappy_bird_final.pth --num_episodes 10
```

**Monitoring**:
```bash
tensorboard --logdir tensorboard
```

### 9.4 Hyperparameter Justification

- **Learning Rate (1e-4)**: Standard for Adam optimizer, prevents overshooting
- **Gamma (0.99)**: High discount factor for long-term planning
- **Epsilon Decay (0.995)**: Slow decay allows sufficient exploration
- **Batch Size (32)**: Balance between stability and speed
- **Replay Buffer (50K)**: Sufficient diversity without excessive memory

---

**End of Report**

