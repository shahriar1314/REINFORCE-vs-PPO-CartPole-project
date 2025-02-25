from train_reinforce import train_reinforce
from train_ppo import train_ppo
from plot_results import plot_results

if __name__ == "__main__":
    rewards_reinforce = train_reinforce()
    rewards_ppo = train_ppo()
    plot_results(rewards_reinforce, rewards_ppo)
