import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make('CartPole-v0')


def compute_fs(rbfss, obs, sig):
    err = rbfss - obs
    err = np.square(err)
    err = np.sum(err, axis=1)
    return np.exp(-err / (2 * sig))


def epsilongreedy(Qf, fvals, epsilonf, num_actions):
    actions = np.dot(Qf, fvals)
    if np.random.random() > epsilonf:
        a = actions.argmax()
    else:
        a = np.random.randint(0, num_actions)
    return a


def Qlookup(Qf, fvals, a):
    actionvals = np.dot(Qf, fvals)
    return actionvals[a]

dim = 4
number_rbfs = 100000
number_actions = 2
epsilon = .1
alpha = .0001
gamma = 0.9
lam = .8

sig = .15
rbfs = np.random.uniform(-1, 1, (number_rbfs, dim))
#rbfs[:, 0] = np.pi/10*rbfs[:, 0]
#rbfs[:, 3] = 2.5*rbfs[:, 3]

Q = np.zeros((number_actions, number_rbfs))

for rounds in range(200):
    eps = epsilon/(rounds + 1)
    elig = np.zeros((number_actions, number_rbfs))
    observation = env.reset()
    observation[0] = observation[0]/(.26) # normalize 15 degrees
    observation[3] = observation[0]/(2.4)

    f_vals = compute_fs(rbfs, observation, sig)
    action = epsilongreedy(Q, f_vals, eps, number_actions)
    for t in range(200):
        #env.render()
        new_observation, reward, done, info = env.step(action)
        new_observation[0] = new_observation[0] / (.26)  # normalize 15 degrees
        new_observation[3] = new_observation[0] / (2.4)
        elig[action, :] = elig[action, :] + f_vals
        f_vals_new = compute_fs(rbfs, new_observation, sig)
        new_action = epsilongreedy(Q, f_vals_new, eps, number_actions)
        Qnewaction = Qlookup(Q, f_vals_new, new_action)

        TDerror = reward + gamma*Qnewaction - Qlookup(Q, f_vals, action)
        #print t
        #print 'TD error: ', TDerror
        #print 'maximum elig. ', elig.max()
        #print 'action:, ', action

        Q[action, :] = Q[action, :] + alpha*TDerror*elig[action, :]

        elig = lam*gamma*elig
        f_vals = f_vals_new
        action = new_action
        if done:
            print 'Ended at timestep: %r' % t
            break
    print 'TD error: ', TDerror
    #print 'maximum elig. ', elig.max()
    #print 'Eligibility : %r ' % elig
    #print 'Q value max: %r ' % Q.max()

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
