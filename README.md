# CartPole Control Based on Reinforcement Learning

## Project Introduction

This project implements and compares multiple control methods on the CartPole-v1 environment.

Methods include:

- Random Policy
- PID Controller
- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Network)

The objective is to keep the pole balanced while maximizing cumulative reward.

---

## Environment

### Operating System
- Windows 11
- WSL2 (Windows Subsystem for Linux)
- Ubuntu 24.04.4 LTS (Codename: noble)

### Python
- Python 3.12.3

### Main Libraries
- Gymnasium v1.3.0
- Stable-Baselines3 v2.9.0
- PyTorch v2.13.0
- NumPy
- Pandas
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
src/
    train_ppo.py
    test_ppo.py
    train_dqn.py
    test_dqn.py
    pid_controller.py
    gamma_experiment.py
    compare_all_methods.py

models/
    ppo_cartpole.zip
    dqn_cartpole.zip

results/
    reward_curve.png
    gamma_compare.png
    survival_compare.png
    reward_compare.png
    theta_compare.png
    x_compare.png
    final_comparison.csv
```

---

## Reward Function

Custom reward:

```python
reward = (
    10
    - 5 * theta**2
    - 0.1 * x**2
    - 0.01 * theta_dot**2
)

if terminated:
    reward -= 100
```

---

## Experimental Results

Final comparison:

| Method | Avg Steps |
|----------|----------:|
| Random | 23 |
| DQN | 221 |
| PPO | 457 |
| PID | 495 |

Observation:

- PPO significantly outperforms DQN.
- PID achieves the best performance in this simplified CartPole task.
- Reinforcement learning demonstrates strong adaptive control capability.

---

## Parameter Analysis

Gamma comparison:

- gamma = 0.90
- gamma = 0.95
- gamma = 0.99

Result:

Higher gamma generally improves long-term control performance.

---

## Demonstration

Control demonstration video:

- PPO controller

---

## Author

East China University of Science and Technology

Zhang Dongxing
