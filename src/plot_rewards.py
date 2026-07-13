import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "../results/training_log.csv"
)

plt.figure(figsize=(10, 6))

plt.plot(df["reward"])

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.title(
    "PPO Training Reward Curve"
)

plt.grid(True)

plt.savefig(
    "../results/reward_curve.png",
    dpi=300
)

plt.show()

print("奖励曲线已保存")