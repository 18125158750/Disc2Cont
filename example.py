# example.py
import gymnasium as gym
import gym_pusht
import numpy as np

env = gym.make("gym_pusht/PushT-v0", render_mode="human", obs_type="environment_state_agent_pos")
observation, info = env.reset()
print(observation)

for _ in range(1000):
    action = np.array([observation[0]+5,200])
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()