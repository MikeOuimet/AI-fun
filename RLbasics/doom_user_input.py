import numpy as np
import gym
import gym_pull



env = gym.make('ppaquette/DoomBasic-v0')


nskips = 10
num_steps = 10
observation = env.reset()


for steps in range(num_steps):
    action_vec = np.zeros(np.shape(env.action_space))
    env.render()
    action = input('Enter your action choice: ')
    print action
    print type(action)
    action_vec[action] = 1
    print action_vec
    for skips in range(nskips):
        observation, reward, done, info = env.step(action_vec)
        env.render()


 