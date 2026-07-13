import os
import gymnasium as gym
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


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


# =========================
# 记录奖励
# =========================
class RewardLogger(BaseCallback):

    def __init__(self):
        super().__init__()

        self.rewards = []

    def _on_step(self):

        if len(self.model.ep_info_buffer) > 0:

            self.rewards.append(
                self.model.ep_info_buffer[-1]["r"]
            )

        return True
    

# =========================
# 环境构建
# =========================
def make_env():

    env = gym.make("CartPole-v1")

    env = RewardWrapper(env)

    env = Monitor(env)

    return env


# =========================
# 主程序
# =========================
if __name__ == "__main__":

    os.makedirs("../models", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    env = make_env()

    model = PPO(
        "MlpPolicy",            # 神经网络类型
        env,                    # 训练环境
        verbose=1,              # 打印日志级别（1表示打印训练中间数据）
        learning_rate=3e-4,     # 学习率（神经网络更新的步长）
        n_steps=2048,           # 采样窗口
        batch_size=64,          # 每次更新时喂给网络的数据量
        gamma=0.99,             # 折扣因子(考虑长期收益)
        device="cpu"
    )

    logger = RewardLogger()

    print("开始训练 PPO...")

    # 让AI在环境里总共交互 50000 步
    model.learn(
        total_timesteps=50000,
        callback=logger
    )

    print("训练完成")

    # 训练结束后，将网络权重保存到本地文件
    model.save("../models/ppo_cartpole")

    print("模型已保存")

    df = pd.DataFrame(
        {
            "reward": logger.rewards
        }
    )

    df.to_csv(
        "../results/training_log.csv",
        index=False
    )

    print("训练日志已保存")