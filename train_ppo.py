import torch
import torch.optim as optim
import gym
from models import PPOActorCritic
from utils import compute_gae
from config import PPO_CONFIG

def train_ppo(env_name="CartPole-v1", num_episodes=PPO_CONFIG['num_episodes']):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = PPOActorCritic(obs_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=PPO_CONFIG['lr'])
    rewards_history = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):  # Check if the reset returns a tuple
            obs = obs[0]  # Get the observation part if it's a tuple
        obs = torch.tensor(obs, dtype=torch.float32)  # Convert it to a tensor
        
        log_probs = []
        rewards = []
        values = []
        actions = []
        total_reward = 0
        
        while True:
            action_probs, value = model(obs)
            action = torch.multinomial(action_probs, 1).item()
            
            new_obs, reward, done, _, _ = env.step(action)
            log_prob = torch.log(action_probs[action])
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            actions.append(action)
            total_reward += reward
            
            obs = torch.tensor(new_obs, dtype=torch.float32)  # Convert the new observation
            if done:
                break
        
        returns, advantages = compute_gae(rewards, values, PPO_CONFIG['gamma'], PPO_CONFIG['gae_lambda'])
        
        loss = -sum(log_prob * adv for log_prob, adv in zip(log_probs, advantages))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        rewards_history.append(total_reward)
        print(f"PPO: Episode {episode + 1}, Total Reward: {total_reward}")
    
    env.close()
    return rewards_history
