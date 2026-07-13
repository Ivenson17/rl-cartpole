import os
import csv
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO


# =========================
# 自定义奖励函数
# =========================
class RewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        x = obs[0]
        theta = obs[2]
        theta_dot = obs[3]

        reward = (
            10
            - 5 * theta ** 2
            - 0.1 * x ** 2
            - 0.01 * theta_dot ** 2
        )

        if terminated:
            reward -= 100

        return obs, reward, terminated, truncated, info


# =========================
# 创建环境
# =========================
def make_env():

    env = gym.make("CartPole-v1")

    env = RewardWrapper(env)

    return env


# =========================
# 测试模型
# =========================
def evaluate_model(model, episodes=50):

    env = make_env()

    rewards = []

    steps_list = []

    for _ in range(episodes):

        obs, info = env.reset()

        done = False

        total_reward = 0

        step_count = 0

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True          # 确定性策略
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            step_count += 1

            done = terminated or truncated

        rewards.append(total_reward)

        steps_list.append(step_count)

    env.close()

    return (
        np.mean(rewards),
        np.mean(steps_list)
    )


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    os.makedirs("../results", exist_ok=True)

    gamma_list = [
        0.90,
        0.95,
        0.99
    ]

    results = []

    for gamma in gamma_list:

        print()
        print("=" * 50)

        print(f"开始训练 gamma={gamma}")

        env = make_env()

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=gamma,
            device="cpu"
        )

        model.learn(
            total_timesteps=30000
        )

        avg_reward, avg_steps = evaluate_model(model)

        print()
        print(f"gamma={gamma}")
        print(f"平均奖励={avg_reward:.2f}")
        print(f"平均步数={avg_steps:.2f}")

        results.append([
            gamma,
            avg_reward,
            avg_steps
        ])

        env.close()

    csv_path = "../results/gamma_results.csv"

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "gamma",
                "avg_reward",
                "avg_steps"
            ]
        )

        writer.writerows(results)

    print()
    print("实验完成")
    print(f"结果保存至: {csv_path}")