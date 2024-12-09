import pandas as pd
import matplotlib.pyplot as plt

file_path = 'average_rewards_log_final.csv'

data_df = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))
plt.plot(data_df["Episode"], data_df["Average Reward (Last 100)"], label="Average Reward (Last 100)")

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Average Rewards over Episodes")
plt.legend()
plt.grid(True)

plt.show()
