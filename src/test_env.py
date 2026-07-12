import gymnasium as gym

env = gym.make("CartPole-v1")

obs, info = env.reset()

print("状态空间：")
print(env.observation_space)

print("动作空间：")
print(env.action_space)

print("当前状态：")
print(obs)