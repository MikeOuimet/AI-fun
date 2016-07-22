import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')

states = 4
dim = 2*states
number_actions = 2
epsilon = .5
alpha = .1
gamma = .9
lam = .5
MLR = .01
nrounds = 10000

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

Tlog = np.zeros(nrounds)
for rounds in range(nrounds):
    eps = epsilon/(rounds + 1)
    max_learning_rate = MLR #/(rounds + 1.0)
    elig = np.zeros((number_actions, dim))
    observation = env.reset()

    ob2 = np.zeros(dim)
    for i in range(states):
        ob2[i]= observation[i]
        ob2[i+states] = 1.0/(observation[i] + 1.0)


    #print observation
    #print ob2

    observation =ob2


    action = epsilongreedy(Q, observation, eps, number_actions)
    for t in range(1000):
        new_observation, reward, done, info = env.step(action)
        for i in range(states):
            ob2[i] = new_observation[i]
            ob2[i + states] = 1.0 / (new_observation[i] + 1.0)
        new_observation = ob2
        for i in range(dim):
            elig[action, i] = elig[action, i] + observation[i]
        new_action = epsilongreedy(Q, new_observation, eps, number_actions)
        Qnewaction = Qlookup(Q, new_observation, new_action)
        #print Qnewaction

        TDerror = reward + gamma * Qnewaction - Qlookup(Q, observation, action)
        #print t
        #print 'TD error: ', TDerror
        # print 'maximum elig. ', elig.max()
        # print 'action:, ', action

        for i in range(dim):
            delta = alpha* TDerror * elig[action, i]
            if delta > max_learning_rate:
                delta = max_learning_rate
            elif delta < -max_learning_rate:
                delta = -max_learning_rate
            Q[action, i] = Q[action, i] + delta

        elig = lam * gamma * elig
        observation = new_observation
        action = new_action
        if done:
            #print 'Ended at timestep: %r' % t
            Tlog[rounds] = t
            #print observation
            break
    if rounds % 1000 ==0:
        print rounds
        #print 'TD error: ', TDerror
        print 'Q : ', Q
        print 'Q values for state', Qlookup(Q, observation, 0), Qlookup(Q, observation, 1)
        #print 'delta params', alpha* TDerror * elig[action, :]
        #print 'elig: ', elig
    #print new_observation

#print Tlog
plt.plot(Tlog)
plt.show()

