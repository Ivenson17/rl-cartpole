import gymnasium as gym
import numpy as np


class RewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        x = obs[0]
        x_dot = obs[1]
        theta = obs[2]
        theta_dot = obs[3]

        # 奖励函数
        reward = (
            10
            - 5 * theta ** 2
            - 0.1 * x ** 2
            - 0.01 * theta_dot ** 2
        )

        if terminated:
            reward -= 100

        return obs, reward, terminated, truncated, info


def make_env(render_mode=None):

    env = gym.make(
        "CartPole-v1",
        render_mode=render_mode
    )

    env = RewardWrapper(env)

    return env


if __name__ == "__main__":

    env = make_env(render_mode="human")

    obs, info = env.reset()

    total_reward = 0

    while True:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(
            f"x={obs[0]:.3f}, "
            f"theta={obs[2]:.3f}, "
            f"reward={reward:.3f}"
        )

        if terminated or truncated:

            print("Episode Reward:", total_reward)

            total_reward = 0

            obs, info = env.reset()