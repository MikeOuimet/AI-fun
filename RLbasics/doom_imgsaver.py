import gym
import gym_pull

env = gym.make('ppaquette/DoomBasic-v0')

num_episodes = 20
for i_episode in range(num_episodes):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
