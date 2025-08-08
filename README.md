# Deep Q-Learning for Atari Games AE4350 Project

## Project Overview

This repository contains an implementation of a Deep Q-Network (DQN) agent for playing Atari 2600 games, based on the CleanRL framework. The focus is on training and evaluating the agent on environments like `BreakoutNoFrameskip-v4`, `PongNoFrameskip-v4`, and `BeamRiderNoFrameskip-v4`. Key modifications include a reduced replay buffer size of 300,000 due to GPU constraints, along with sensitivity analysis on hyperparameters such as learning rate, discount factor, and batch size.

This project was developed as part of the AE4350 assignment on bio-inspired intelligence, aiming to demonstrate DQN's application in high-dimensional, pixel-based reinforcement learning tasks. Results include learning curves, performance metrics, and qualitative gameplay videos.

## Features

- Single-file DQN implementation adapted from CleanRL.
- Preprocessing for Atari environments (frame stacking, grayscale conversion, reward clipping).
- Sensitivity analysis on key hyperparameters.
- TensorBoard logging for metrics visualization.
- Gameplay video recordings for qualitative evaluation.

## Demo Videos

https://github.com/user-attachments/assets/375b4a2c-44ee-42b1-b5af-102bf84453ee

https://github.com/user-attachments/assets/47a3f5a1-7cb7-4711-9cb4-b1106d7386b3

https://github.com/user-attachments/assets/1dee3c6e-48e0-4293-84b6-5026c6e8da87

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended, but code runs on CPU with limitations)
- pip or uv for package management

### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/atari-dqn.git
   cd atari-dqn
   ```

2. Install dependencies:
   ```
   uv pip install -r requirements.txt
   ```
   (Or use `pip install -r requirements.txt` if not using uv.)

   Example `requirements.txt` (included in repo):
   ```
   gymnasium[atari,accept-rom-license]
   torch
   numpy
   tensorboard
   ```

## Usage

### Training the Agent
Run the DQN script with default parameters:
```
python attari_dqn.py --env_id BreakoutNoFrameskip-v4
```
- Key arguments (modifiable via command line):
  - `--env_id`: Environment name (e.g., `BreakoutNoFrameskip-v4`).
  - `--buffer_size`: Replay buffer size (default: 300000 due to hardware limits).
  - `--total_timesteps`: Training steps (default: 1000000).
  - `--learning_rate`: Optimizer learning rate (default: 1e-4).
  - `--capture_video`: Set to True to record gameplay (outputs to /videos).

For sensitivity analysis examples:
- Low learning rate: `python attari_dqn.py --learning_rate 2.5e-5`
- Smaller batch: `python attari_dqn.py --batch_size 16`

### Viewing Logs
Launch TensorBoard to visualize metrics:
```
tensorboard --logdir runs/
```
Access at `http://localhost:6006`.


## Experiments and Results

### Methodology
The DQN implementation uses convolutional layers for Q-value approximation from stacked frames. Training involves epsilon-greedy exploration, experience replay, and target network updates. Due to GPU limitations, buffer size is capped at 300,000, impacting sample diversity in sparse-reward games.

### Sensitivity Analysis
We varied one parameter per run (buffer fixed at 300,000):
- Baseline: Defaults.
- lowLR: learning_rate=2.5e-5.
- highLR: learning_rate=2.5e-4.
- discount09: gamma=0.90.
- batch16: batch_size=16.

Results show sensitivity to learning rate and gamma, with higher rates causing instability.

### Key Results
- **BeamRider**: Strong learning with returns >5000.
- **Breakout**: Moderate improvement, plateauing at ~55.
- **Pong**: Limited progress, returns ~0-13.

See `/figures/charts/` for TensorBoard plots.

## Folder Structure

- `attari_dqn.py`: Main DQN training script.
- `requirements.txt`: Dependencies.
- `/runs/`: TensorBoard logs.
- `/figures/charts/`: Result plots (e.g., ep_return_all.png).
- `/videos/`: Gameplay recordings (MP4 files).
- `README.md`: This file.


## Acknowledgments and References

- Based on [CleanRL](https://docs.cleanrl.dev/rl-algorithms/dqn/).
- Original DQN: Mnih et al. (2015), "Human-level control through deep reinforcement learning," Nature.
- Atari environments from Gymnasium.

License: MIT
