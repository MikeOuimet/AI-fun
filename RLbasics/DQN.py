# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 21:36:05 2016

@author: mike
"""
import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make('Pong-v0')

def reduce_image(_obs):
    new_obs = np.sum(_obs, 2) / (3. * 256.)
    new_obs[new_obs < .5] = 0
    new_obs[new_obs >= .5] = 1
    return new_obs

env.reset()
for i_episode in range(1):
    observation = env.reset()
    done = False
    observation = reduce_image(observation)
    print np.shape(observation)
    plt.pcolormesh(observation)
'''    while not done:
        observation = reduce_image(observation)
        print observation
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        #print done
env.render(close=True)
'''