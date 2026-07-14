import gymnasium as gym


# =========================
# PID参数
# =========================
Kx = -0.5
Kv = -0.5
Kp = 50
Kd = 10


# =========================
# 创建环境
# =========================
env = gym.make(
    "CartPole-v1",
    render_mode="human"
)

episodes = 3


# =========================
# 测试
# =========================
for episode in range(episodes):

    obs, info = env.reset()

    done = False

    total_steps = 0

    while not done:

        x = obs[0]
        x_dot = obs[1]
        theta = obs[2]
        theta_dot = obs[3]

        u = (
            Kx * x
            + Kv * x_dot
            + Kp * theta
            + Kd * theta_dot
        )

        if u > 0:
            action = 1
        else:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_steps += 1

    print(
        f"Episode {episode+1}: "
        f"{total_steps} steps"
    )

env.close()