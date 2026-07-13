import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "../results/gamma_results.csv"
)

plt.figure(figsize=(8,5))

plt.bar(
    df["gamma"].astype(str),
    df["avg_steps"]
)

plt.xlabel("Gamma")

plt.ylabel("Average Survival Steps")

plt.title(
    "Effect of Gamma on PPO Performance"
)

plt.grid(axis="y")

plt.savefig(
    "../results/gamma_compare.png",
    dpi=300
)

plt.show()

print("图像已保存")