import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')

dim = 4
number_rbfs = 100

def rbf(x, y, sig):
    ans = np.exp(-1)
    return ans

sig = .2
rbfs = np.random.uniform(-1, 1, (number_rbfs, dim))
rbfs[:, 0] = np.pi/4*rbfs[:, 0]
rbfs[:, 3] = 2.5*rbfs[:, 3]

observation = env.reset()
env.render()
print(observation)
action = env.action_space.sample()
observation, reward, done, info = env.step(action)

print 'The action is %r and the reward is %r' % (action, reward)
print 'Done: %r, observation: %r' % (done, observation)

err = rbfs - observation
err = np.square(err)
err = np.sum(err, axis=1)
vals = np.exp(-err/(2*sig))

print vals.max()



#for i_episode in range(20):
#    observation = env.reset()
#    for t in range(100):
#        env.render()
#        print(observation)
#        action = env.action_space.sample()
#        observation, reward, done, info = env.step(action)
#        if done:
#            print("Episode finished after {} timesteps".format(t+1))
#            break

