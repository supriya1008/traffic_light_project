import gym
import numpy as np
from gym import spaces

class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()

        # Observation space: [vehicle_count, emergency_detected (0/1)]
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([100, 1]), dtype=np.float32)

        # Action space: 0 = Reduce Red, 1 = Keep Same, 2 = Increase Red
        self.action_space = spaces.Discrete(3)

        self.state = np.array([50, 0])  # Initial state
        self.emergency_detected = False

    def step(self, action):
        vehicle_count, emergency = self.state

        # Adjust red signal duration based on action
        if action == 0:
            red_time = max(20, vehicle_count - 10)
        elif action == 2:
            red_time = min(60, vehicle_count + 10)
        else:
            red_time = vehicle_count  # Keep same

        # Simulate next traffic state
        vehicle_count = max(0, min(100, vehicle_count + np.random.randint(-10, 10)))
        self.state = np.array([vehicle_count, emergency])

        # Reward function: Less congestion is better, emergency gets priority
        reward = -vehicle_count
        if emergency:
            reward += 50

        return self.state, reward, False, {}

    def reset(self):
        self.state = np.array([np.random.randint(10, 100), 0])
        return self.state

    def set_emergency(self, detected):
        self.state[1] = 1 if detected else 0  # Set emergency flag
