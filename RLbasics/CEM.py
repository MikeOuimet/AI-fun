import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')

n_samples = 100
dim = max(np.shape(env.observation_space))
#dim_actions = env.action_space.n
elite = .2

mean = np.zeros(dim)
cov = np.eye(dim)
samples = np.random.multivariate_normal(mean, cov, n_samples)

print samples.shape

for n_cem in range(10):
    score = np.zeros(n_samples)
    nrounds = 10
    for samp in range(n_samples):
        count = 0
        for rounds in range(nrounds):
            observation = env.reset()
            for time in range(1500):
                if np.dot(samples[samp, :], observation) > 0:
                    action = 0
                else:
                    action = 1
                observation, reward, done, info = env.step(action)

                if done:
                    break
            #print time
            count += time
        score[samp]=count/(nrounds + 0.0)
    avg = score.mean()
    print 'avg is ', avg
    sorted = score.argsort()
    new_samp = np.zeros((20,4))
    for i in range(20):
        new_samp[i, :] = samples[sorted[-i], :]

    mean = new_samp.mean(axis=0)
    std = new_samp.std(axis=0)
    cov = np.square(std * np.eye(4))
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    #print mean
    #print cov
