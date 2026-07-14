import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DQN


# =========================
# 奖励函数
# =========================
def custom_reward(obs, terminated):

    x = obs[0]
    theta = obs[2]
    theta_dot = obs[3]

    reward = (
        10
        - 5 * theta**2
        - 0.1 * x**2
        - 0.01 * theta_dot**2
    )

    if terminated:
        reward -= 100

    return reward


# =========================
# Random
# =========================
def evaluate_random(episodes=50):

    env = gym.make("CartPole-v1")

    step_list = []
    reward_list = []
    theta_list = []
    x_list = []

    for _ in range(episodes):

        obs, info = env.reset()

        done = False

        total_reward = 0
        total_theta = 0
        total_x = 0
        steps = 0

        while not done:

            action = env.action_space.sample()

            obs, _, terminated, truncated, info = env.step(action)

            reward = custom_reward(obs, terminated)

            total_reward += reward

            total_theta += abs(obs[2])
            total_x += abs(obs[0])

            steps += 1

            done = terminated or truncated

        step_list.append(steps)
        reward_list.append(total_reward)
        theta_list.append(total_theta / steps)
        x_list.append(total_x / steps)

    env.close()

    return (
        np.mean(step_list),
        np.mean(reward_list),
        np.mean(theta_list),
        np.mean(x_list)
    )


# =========================
# PID
# =========================
def evaluate_pid(episodes=50):

    env = gym.make("CartPole-v1")

    Kx = -0.5
    Kv = -0.5
    Kp = 50
    Kd = 10

    step_list = []
    reward_list = []
    theta_list = []
    x_list = []

    for _ in range(episodes):

        obs, info = env.reset()

        done = False

        total_reward = 0
        total_theta = 0
        total_x = 0
        steps = 0

        while not done:

            x = obs[0]
            x_dot = obs[1]
            theta = obs[2]
            theta_dot = obs[3]

            u = (
                Kx*x
                + Kv*x_dot
                + Kp*theta
                + Kd*theta_dot
            )

            action = 1 if u > 0 else 0

            obs, _, terminated, truncated, info = env.step(action)

            reward = custom_reward(obs, terminated)

            total_reward += reward

            total_theta += abs(obs[2])
            total_x += abs(obs[0])

            steps += 1

            done = terminated or truncated

        step_list.append(steps)
        reward_list.append(total_reward)
        theta_list.append(total_theta / steps)
        x_list.append(total_x / steps)

    env.close()

    return (
        np.mean(step_list),
        np.mean(reward_list),
        np.mean(theta_list),
        np.mean(x_list)
    )


# =========================
# PPO
# =========================
def evaluate_ppo(episodes=50):

    env = gym.make("CartPole-v1")

    model = PPO.load("../models/ppo_cartpole")

    step_list = []
    reward_list = []
    theta_list = []
    x_list = []

    for _ in range(episodes):

        obs, info = env.reset()

        done = False

        total_reward = 0
        total_theta = 0
        total_x = 0
        steps = 0

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, _, terminated, truncated, info = env.step(action)

            reward = custom_reward(obs, terminated)

            total_reward += reward

            total_theta += abs(obs[2])
            total_x += abs(obs[0])

            steps += 1

            done = terminated or truncated

        step_list.append(steps)
        reward_list.append(total_reward)
        theta_list.append(total_theta / steps)
        x_list.append(total_x / steps)

    env.close()

    return (
        np.mean(step_list),
        np.mean(reward_list),
        np.mean(theta_list),
        np.mean(x_list)
    )


# =========================
# DQN
# =========================
def evaluate_dqn(episodes=50):

    env = gym.make("CartPole-v1")

    model = DQN.load("../models/dqn_cartpole")

    step_list = []
    reward_list = []
    theta_list = []
    x_list = []

    for _ in range(episodes):

        obs, info = env.reset()

        done = False

        total_reward = 0
        total_theta = 0
        total_x = 0
        steps = 0

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, _, terminated, truncated, info = env.step(action)

            reward = custom_reward(obs, terminated)

            total_reward += reward

            total_theta += abs(obs[2])
            total_x += abs(obs[0])

            steps += 1

            done = terminated or truncated

        step_list.append(steps)
        reward_list.append(total_reward)
        theta_list.append(total_theta / steps)
        x_list.append(total_x / steps)

    env.close()

    return (
        np.mean(step_list),
        np.mean(reward_list),
        np.mean(theta_list),
        np.mean(x_list)
    )


# =========================
# 主程序
# =========================

if __name__ == "__main__":

    methods = []
    steps_data = []
    rewards_data = []
    theta_data = []
    x_data = []

    print("Testing Random...")
    r = evaluate_random()

    print("Testing PID...")
    p = evaluate_pid()

    print("Testing PPO...")
    o = evaluate_ppo()

    print("Testing DQN...")
    d = evaluate_dqn()

    results = {
        "Random": r,
        "PID": p,
        "PPO": o,
        "DQN": d
    }

    for name, data in results.items():

        methods.append(name)

        steps_data.append(data[0])
        rewards_data.append(data[1])
        theta_data.append(data[2])
        x_data.append(data[3])

    df = pd.DataFrame({
        "Method": methods,
        "Steps": steps_data,
        "Reward": rewards_data,
        "ThetaError": theta_data,
        "XError": x_data
    })

    df.to_csv(
        "../results/final_comparison.csv",
        index=False
    )

    print(df)

    metrics = {
        "survival_compare.png":
            ("Average Steps", steps_data),

        "reward_compare.png":
            ("Average Reward", rewards_data),

        "theta_compare.png":
            ("Mean |Theta|", theta_data),

        "x_compare.png":
            ("Mean |X|", x_data)
    }

    for filename, (ylabel, values) in metrics.items():

        plt.figure(figsize=(8,5))

        plt.bar(methods, values)

        plt.ylabel(ylabel)

        plt.title(ylabel)

        plt.grid(axis="y")

        plt.savefig(
            f"../results/{filename}",
            dpi=300
        )

        plt.close()

    print()
    print("所有结果已保存到 results/")