import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')

dim = 4
number_actions = 2
epsilon = .05
alpha = .005
gamma = .9
lam = .9


def epsilongreedy(Qf, obs, epsilonf, num_actions):
    actions = np.dot(Qf, obs)
    if np.random.random() > epsilonf:
        a = actions.argmax()
    else:
        a = np.random.randint(0, num_actions)
    return a


def Qlookup(Qf, obs, a):
    actionvals = np.dot(Qf, obs)
    return actionvals[a]


Q = np.zeros((number_actions, dim))

nrounds = 1000
Tlog = np.zeros(nrounds)
for rounds in range(nrounds):
    eps = epsilon/(rounds + 1)
    elig = np.zeros((number_actions, dim))
    observation = env.reset()


    action = epsilongreedy(Q, observation, eps, number_actions)
    for t in range(200):
        new_observation, reward, done, info = env.step(action)
        #new_observation[0] = 10*new_observation[0]
        #new_observation[3]=new_observation[3]/(2.4)
        for i in range(dim):
            elig[action, i] = elig[action, i] + observation[i]
        new_action = epsilongreedy(Q, new_observation, eps, number_actions)
        Qnewaction = Qlookup(Q, new_observation, new_action)
        #print Qnewaction

        TDerror = reward + gamma * Qnewaction - Qlookup(Q, observation, action)
        # print t
        #print 'TD error: ', TDerror
        # print 'maximum elig. ', elig.max()
        # print 'action:, ', action

        for i in range(dim):
            Q[action, i] = Q[action, i] + alpha/(nrounds+1) * TDerror * elig[action, i]

        elig = lam * gamma * elig
        observation = new_observation
        action = new_action
        if done:
            #print 'Ended at timestep: %r' % t
            Tlog[rounds] = t
            #print observation
            break
    if rounds % 1000 ==0:
        print t
        print 'TD error: ', TDerror
        print 'Q : ', Q
        #print 'elig: ', elig
    #print new_observation

#print Tlog
plt.plot(Tlog)
plt.show()