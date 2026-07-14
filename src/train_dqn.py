import os
import gymnasium as gym

from stable_baselines3 import DQN


# =========================
# 自定义奖励函数
# =========================
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


# =========================
# 环境
# =========================
def make_env():

    env = gym.make("CartPole-v1")

    env = RewardWrapper(env)

    return env


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    os.makedirs("../models", exist_ok=True)

    env = make_env()

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,                 # 学习率
        gamma=0.99,                         # 折扣因子
        buffer_size=100000,                 # 经验回放缓冲区大小
        learning_starts=1000,               # 开始学习的步数
        batch_size=64,                      # 批次大小
        target_update_interval=500,         # 目标网络更新频率
        exploration_fraction=0.2,           # 探索阶段占总步数的比例
        exploration_final_eps=0.05,         # 探索阶段结束后的最终随机率
        device="cpu"
    )

    print("开始训练 DQN...")

    model.learn(
        total_timesteps=50000
    )

    print("训练完成")

    model.save(
        "../models/dqn_cartpole"
    )

    print("模型已保存")