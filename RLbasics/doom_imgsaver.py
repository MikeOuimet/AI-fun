import numpy as np
import gym
import gym_pull

env = gym.make('ppaquette/DoomBasic-v0')

num_episodes = 2
episode_length = 10
height = 480
width = 640



print env.action_space
print env.observation_space

observation = env.reset()
action = env.action_space.sample()
print action

observation_vec = np.zeros(shape =(num_episodes, episode_length, height, width, 3), dtype='uint8')
for i_episode in range(num_episodes):
	print i_episode
	observation = env.reset()
	for t in range(episode_length):
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		observation_vec[i_episode, t, :, :, :] = observation
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
np.save(file='observations', arr=observation_vec)
