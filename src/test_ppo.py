import gymnasium as gym
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
# 加载模型
# =========================
model = PPO.load(
    "../models/ppo_cartpole"
)


# =========================
# 创建图形环境
# =========================
env = gym.make(
    "CartPole-v1",
    render_mode="human"
)

env = RewardWrapper(env)


# =========================
# 测试
# =========================
episodes = 3

for episode in range(episodes):

    obs, info = env.reset()

    done = False

    total_reward = 0

    total_steps = 0

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += reward

        total_steps += 1

    print(
        f"Episode {episode+1}"
    )

    print(
        f"Steps : {total_steps}"
    )

    print(
        f"Reward: {total_reward:.2f}"
    )

    print("-" * 30)

env.close()