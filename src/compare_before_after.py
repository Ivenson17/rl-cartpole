import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO


# ==================================
# 自定义奖励函数
# ==================================
class RewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        x = obs[0]
        x_dot = obs[1]
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

        return obs, reward, terminated, truncated, info


# ==================================
# Random
# ==================================
def test_random_policy(episodes=100):

    env = RewardWrapper(
        gym.make("CartPole-v1")
    )

    episode_lengths = []
    episode_rewards = []

    for _ in range(episodes):

        obs, info = env.reset()

        done = False

        step_count = 0
        total_reward = 0

        while not done:

            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            step_count += 1
            total_reward += reward

        episode_lengths.append(step_count)
        episode_rewards.append(total_reward)

    env.close()

    return (
        np.mean(episode_lengths),
        np.mean(episode_rewards)
    )


# ==================================
# PPO
# ==================================
def test_ppo_policy(episodes=100):

    env = RewardWrapper(
        gym.make("CartPole-v1")
    )

    model = PPO.load(
        "../models/ppo_cartpole"
    )

    episode_lengths = []
    episode_rewards = []

    for _ in range(episodes):

        obs, info = env.reset()

        done = False

        step_count = 0
        total_reward = 0

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            step_count += 1
            total_reward += reward

        episode_lengths.append(step_count)
        episode_rewards.append(total_reward)

    env.close()

    return (
        np.mean(episode_lengths),
        np.mean(episode_rewards)
    )


# ==================================
# 主程序
# ==================================
if __name__ == "__main__":

    print("测试 Random")

    random_steps, random_rewards = (
        test_random_policy()
    )

    print(
        f"平均存活步数: {random_steps:.2f}"
    )

    print(
        f"平均累计奖励: {random_rewards:.2f}"
    )

    print()

    print("测试 PPO")

    ppo_steps, ppo_rewards = (
        test_ppo_policy()
    )

    print(
        f"平均存活步数: {ppo_steps:.2f}"
    )

    print(
        f"平均累计奖励: {ppo_rewards:.2f}"
    )

    # ==========================
    # 图1:存活步数
    # ==========================

    plt.figure(figsize=(8,5))

    plt.bar(
        ["Random", "PPO"],
        [random_steps, ppo_steps]
    )

    plt.ylabel("Average Survival Steps")

    plt.title(
        "Before vs After Training (Steps)"
    )

    plt.grid(axis="y")

    plt.savefig(
        "../results/before_after_steps.png",
        dpi=300
    )

    # ==========================
    # 图2:累计奖励
    # ==========================

    plt.figure(figsize=(8,5))

    plt.bar(
        ["Random", "PPO"],
        [random_rewards, ppo_rewards]
    )

    plt.ylabel("Average Reward")

    plt.title(
        "Before vs After Training (Reward)"
    )

    plt.grid(axis="y")

    plt.savefig(
        "../results/before_after_rewards.png",
        dpi=300
    )

    plt.show()

    print()
    print("结果已保存")