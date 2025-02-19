import gymnasium as gym
import gym_pusht
import pygame
import numpy as np

if __name__ == "__main__":
    env = gym.make("gym_pusht/PushT-v0", render_mode="human") # Or any other PushT env
    observation, info = env.reset()
    pygame.init() # Initialize pygame for event handling
    clock = pygame.time.Clock()
    terminated = False
    truncated = False

    action_space = [
        [20, 0],  # Right (Action 0, Right Arrow)
        [0, -20],  # Up (Action 1, Up Arrow)
        [-20, 0], # Left (Action 2, Left Arrow)
        [0, 20]  # Down (Action 3, Down Arrow)
    ]

    while not terminated and not truncated:
        env.render() # Render the environment

        action = None # Initialize action to None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True # Exit if window close button is pressed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    action_index = 0
                    action = action_space[action_index]
                    print("Action: Right")
                elif event.key == pygame.K_UP:
                    action_index = 1
                    action = action_space[action_index]
                    print("Action: Up")
                elif event.key == pygame.K_LEFT:
                    action_index = 2
                    action = action_space[action_index]
                    print("Action: Left")
                elif event.key == pygame.K_DOWN:
                    action_index = 3
                    action = action_space[action_index]
                    print("Action: Down")
                elif event.key == pygame.K_q:
                    terminated = True # Exit if 'q' is pressed

        if action is not None:
            observation, reward, terminated, truncated, info = env.step((observation[:2] + np.array(action)).astype(np.float32)) # Apply action to xy position
            print(f"Observation: {observation}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            print(f"is_success: {info['is_success']}")
            # print(f"Info: {info}")
            if terminated or truncated:
                print("Episode finished. Resetting environment.")
                observation, info = env.reset()

        clock.tick(30) # Limit frame rate to 30 FPS

    env.close()
    pygame.quit()