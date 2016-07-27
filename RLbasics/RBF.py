import numpy as np
import gym
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
env = gym.make('MountainCar-v0')


def compute_fs(rbfss, obs, sig):
    err = rbfss - obs
    err = np.square(err)
    err = np.sum(err, axis=1)
    return np.exp(-err / (2 * sig**2))


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

def visualize_v(rbfs, Q, flag):
    activations = np.zeros((len(rbfs),len(rbfs)))
    Vplot = np.zeros(len(rbfs))
    for point in range(len(rbfs)):
        activations[point, :] =  compute_fs(rbfs, rbfs[point], sig)
        interim = []
        for a in range(number_actions):
            interim.append(Qlookup(Q, activations[point, :], a))
        
        Vplot[point] = np.max(interim)
        
    if flag:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(rbfs[:,0], rbfs[:,1], Vplot)
        plt.xlabel('position')
        plt.ylabel('velocity')
        plt.show()
    #else:
    #    fig = plt.figure()
    #    plt.pcolor(rbfs[:,0], rbfs[:,1], Vplot)
    #    plt.xlabel('position')
    #    plt.ylabel('velocity')
    #    plt.show()
        #X,Y = np.mesh


dim = 2
number_rbfs = 400
number_actions = 3
eps = .1
final_eps = .001
NUM_EPISODES = 5000
NUM_STEPS = 200
decayrate = (final_eps/eps)**(1/(NUM_EPISODES+0.0))

alpha = .01
gamma = 0.99
lam = .5
scatter_flag = 1


sig = 1.0/(np.sqrt(number_rbfs)-1.0)
#rbfs = np.random.uniform(-1, 1, (number_rbfs, dim))
rbfs = np.ones((number_rbfs, dim))
xpoints= np.linspace(-1,1, np.sqrt(number_rbfs))

count = 0
for x in xpoints:
    for y in xpoints:
        rbfs[count,:] = [x,y]
        count +=1

print decayrate
f_vals_done = np.zeros((number_rbfs))

#print env.observation_space.low
#print env.observation_space.high
#print env.action_space

Q = -10*np.ones((number_actions, number_rbfs))
new_action = 0
donedone = False
count = 0



countvec = np.zeros(NUM_EPISODES)
for rounds in range(NUM_EPISODES):
    eps = eps*decayrate
    elig = np.zeros((number_actions, number_rbfs))
    observation = env.reset()
    observation[0] = (observation[0] + .3)/(.9)
    observation[1] = 12*observation[1]
    f_vals = compute_fs(rbfs, observation, sig)
    action = epsilongreedy(Q, f_vals, eps, number_actions)
    for t in range(NUM_STEPS):
        #if rounds % 1000 ==0:
        #    env.render()
        new_observation, reward, done, info = env.step(action)
        new_observation[0] = (new_observation[0] + .3)/(.9)
        new_observation[1] = 12*new_observation[1]
        elig[action, :] = f_vals
        #print 'maximum eligibilities', elig.max()
        f_vals_new = compute_fs(rbfs, new_observation, sig)     
        
        new_action = epsilongreedy(Q, f_vals_new, eps, number_actions)
        Qnewaction = Qlookup(Q, f_vals_new, new_action)

        TDerror = reward + gamma*Qnewaction - Qlookup(Q, f_vals, action)
        if done:
            TDerror = reward  - Qlookup(Q, f_vals, action)
        
        Qbefore = Qlookup(Q, f_vals, action)

        Q[action, :] = Q[action, :] + alpha*TDerror*elig[action, :]
        #if done:
        #    print 'Q after', Qlookup(Q, f_vals, action)
        
        TDerrornew = reward + gamma*Qnewaction - Qlookup(Q, f_vals, action)
        if done:
            TDerrornew = reward - Qlookup(Q, f_vals, action)
        Qafter = Qlookup(Q, f_vals, action)
        
        #TDerror - TDerrornew 
        
        if done and rounds % 100 == 0:
            print 'time:', t
            print 'observation', new_observation
            print 'reward:', reward
            #print 'Target: ', reward + gamma*Qnewaction
            #print 'max TD update', np.abs(alpha*TDerror*elig[action, :]).max()
            #print 'min TD update', np.abs(alpha*TDerror*elig[action, :]).min()
            print 'max eligibility', elig[action, :].max()
            print 'TD error before', TDerror
            print 'TD error after', TDerrornew
            print 'Q before', Qbefore
            print 'Q after', Qafter
            print ''
        elig[:] = lam*gamma*elig
        f_vals[:] = f_vals_new
        action = new_action
        if done:
            count +=1
            #print 'Ended at timestep: %r' % t
            #print 'Fraction completed:', count/(rounds +0.0)
            donedone= True
            #f_vals_done[:] = f_vals
            #print ''
            break
        if Qafter > 100:
            print 'Q diverged'
            break
    if done:
        countvec[rounds] = t
    else:
        countvec[rounds] = NUM_STEPS
    if rounds % 100 ==0:
        print rounds
        if rounds >0:
            print 'end Q', 
            print 'Fraction completed:', count/(rounds +0.0)
            visualize_v(rbfs, Q, scatter_flag)
            print ''
            

print 'Fraction completed:', count/(rounds +0.0)

plt.plot(countvec)
plt.show()
