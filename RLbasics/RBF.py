import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make('MountainCar-v0')


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

dim = 2
number_rbfs = 1000
number_actions = 3
eps = .5
alpha = .005
gamma = 0.95
lam = .2

sig = .0025
rbfs = np.random.uniform(-1, 1, (number_rbfs, dim))
f_vals_done = np.zeros((number_rbfs))
#rbfs[:, 0] = np.pi/10*rbfs[:, 0]
#rbfs[:, 3] = 2.5*rbfs[:, 3]

print env.observation_space.low
print env.observation_space.high
print env.action_space

Q = np.ones((number_actions, number_rbfs))
new_action = 0
donedone = False
count = 0
NUM_EPISODES = 5000
NUM_STEPS = 1000
countvec = np.zeros(NUM_EPISODES)
for rounds in range(NUM_EPISODES):
    eps = eps*.999
    elig = np.zeros((number_actions, number_rbfs))
    observation = env.reset()
    #observation[0] = observation[0]/(.2) # normalize 15 degrees
    observation[0] = (observation[0] + .3)
    observation[1] = 12*observation[1]
    #observation[3] = observation[3]/(2.4)
    
    #print 'observation', observation
    f_vals = compute_fs(rbfs, observation, sig)
    #if donedone and rounds % 10 ==0:
    #    print 'Most recent goal Q', np.dot(Q, f_vals_done)
    #    print 'Starting state Q', np.dot(Q, f_vals)
    #    print ''
    #print 'activations', f_vals
    action = epsilongreedy(Q, f_vals, eps, number_actions)
    for t in range(NUM_STEPS):
        if rounds % 100 ==0:
            env.render()
        new_observation, reward, done, info = env.step(action)
        new_observation[0] = (new_observation[0] + .3)
        new_observation[1] = 12*new_observation[1]
        #if done:        
            #print 'observation', new_observation
            #print 'possible action values', np.dot(Q,f_vals)
            #print 'action', action
        
        elig[action, :] = elig[action, :] + f_vals
        #print 'maximum eligibilities', elig.max()
        f_vals_new = compute_fs(rbfs, new_observation, sig)
        
        new_action = epsilongreedy(Q, f_vals_new, eps, number_actions)
        Qnewaction = Qlookup(Q, f_vals_new, new_action)
        #if done:
            #print 'Q of new action', Qnewaction

        TDerror = reward + gamma*Qnewaction - Qlookup(Q, f_vals, action)
        #if done:
        #    TDerror = reward - Qlookup(Q, f_vals, action)
        
        #if done:
        #    print eps
        #    print 'Q before', Qlookup(Q, f_vals, action)
            #print t
            #print 'TD error: ', TDerror
            #print 'maximum elig. ', elig.max()
            #print 'action:, ', action

        Q[action, :] = Q[action, :] + alpha*TDerror*elig[action, :]
        #if done:
        #    print 'Q after', Qlookup(Q, f_vals, action)
        
        TDerrornew = reward + gamma*Qnewaction - Qlookup(Q, f_vals, action)
        #if done:
        #    TDerrornew = reward - Qlookup(Q, f_vals, action)
        
        elig[:] = lam*gamma*elig
        f_vals[:] = f_vals_new
        #if done:
            #print 'reward:', reward
            #print 'Target: ', reward + gamma*Qnewaction
            #print 'max TD update', np.abs(alpha*TDerror*elig[action, :]).max()
            #print 'min TD update', np.abs(alpha*TDerror*elig[action, :]).min()
            #print 'TD error before', TDerror
            #print 'TD error after', TDerrornew
            
        action = new_action
        if done:
            count +=1
            #print 'Ended at timestep: %r' % t
            #print 'Fraction completed:', count/(rounds +0.0)
            donedone= True
            #f_vals_done[:] = f_vals
            #print ''
            break
        if Q.max() > 1000:
            print 'Q diverged'
            break
    if done:
        countvec[rounds] = t
    else:
        countvec[rounds] = NUM_STEPS
    if rounds % 100 ==0:
        print rounds
        if rounds >0:
            print 'Fraction completed:', count/(rounds +0.0)
            print ''

print 'Fraction completed:', count/(rounds +0.0)

plt.plot(countvec)
plt.show()