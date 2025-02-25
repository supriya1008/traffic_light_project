import gym
import torch
from stable_baselines3 import PPO
from traffic_env import TrafficEnv

# Create traffic environment
env = TrafficEnv()

# Initialize PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save trained model
model.save("models/ppo_traffic")
print("✅ Model Trained and Saved!")
