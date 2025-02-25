import torch
import torch.optim as optim
import gym
from models import PolicyNetwork
from utils import discount_rewards
from config import REINFORCE_CONFIG



def train_reinforce(env_name="CartPole-v1", num_episodes=REINFORCE_CONFIG['num_episodes']):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=REINFORCE_CONFIG['lr'])
    rewards_history = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()  # Adjusted to unpack the tuple
        obs = torch.tensor(obs, dtype=torch.float32)  # Convert observation to tensor at the beginning
        log_probs = []
        rewards = []
        total_reward = 0
        
        while True:
            action_probs = policy(obs)
            action = torch.multinomial(action_probs, 1).item()
            
            new_obs, reward, done, _, _ = env.step(action)  # Adjusted to match the returned values
            log_prob = torch.log(action_probs[action])
            
            log_probs.append(log_prob) 
            rewards.append(reward)
            total_reward += reward
            
            obs = torch.tensor(new_obs, dtype=torch.float32)  # Convert new observation to tensor
            if done:
                break
        
        returns = discount_rewards(rewards, REINFORCE_CONFIG['gamma'])
        loss = -sum(log_prob * ret for log_prob, ret in zip(log_probs, returns))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        rewards_history.append(total_reward)
        print(f"REINFORCE: Episode {episode + 1}, Total Reward: {total_reward}")
    
    env.close()
    return rewards_history