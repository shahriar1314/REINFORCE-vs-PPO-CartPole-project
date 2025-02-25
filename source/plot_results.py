import matplotlib.pyplot as plt

def plot_results(rewards_reinforce, rewards_ppo):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_reinforce, label="REINFORCE")
    plt.plot(rewards_ppo, label="PPO")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Performance Comparison of REINFORCE and PPO")
    plt.legend()
    plt.show()
