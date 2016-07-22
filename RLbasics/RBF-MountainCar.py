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
number_rbfs = 100
number_actions = 2
epsilon = .05
alpha = .01
gamma = 0.95
lam = .8
sig = .05
rbfs = np.random.uniform(-1.0, 1.0, (number_rbfs, dim))

Q = 100*np.ones((number_actions, number_rbfs))

for rounds in range(4000):
    if rounds % 10 ==0:
        print 'rounds', rounds
    eps = epsilon/(rounds + 1)
    elig = np.zeros((number_actions, number_rbfs))
    observation = env.reset()

    f_vals = compute_fs(rbfs, observation, sig)
    action = epsilongreedy(Q, f_vals, eps, number_actions)
    for t in range(10001):
        if rounds % 100 == 0:
            env.render()
        new_observation, reward, done, info = env.step(action)
        new_observation[0] = .5*new_observation[0]
        new_observation[1] = 15*new_observation[1]
        elig[action, :] = elig[action, :] + f_vals
        f_vals_new = compute_fs(rbfs, new_observation, sig)
        new_action = epsilongreedy(Q, f_vals_new, eps, number_actions)
        Qnewaction = Qlookup(Q, f_vals_new, new_action)
        TDerror = reward + gamma*Qnewaction - Qlookup(Q, f_vals, action)
        if rounds % 100 == 0:
            env.render()
            if t % 200 == 0:
                print 'time', t
                print 'state', new_observation
                print 'TD error: ', TDerror
                print 'Target: ', reward + gamma * Qnewaction
                print 'Q estimate', Qlookup(Q, f_vals, action)
                print ''
        if done:
            print 'Done!'
            break
        Q[action, :] = Q[action, :] + alpha*TDerror*elig[action, :]

        elig = lam*gamma*elig
        f_vals = f_vals_new
        action = new_action



'''
x = np.linspace(-2, 2, 100)
theta = np.linspace(-2, 2, 100)
Qheight = np.zeros((100, 100))
for xx in x:
    for thetatheta in theta:
        print xx, thetatheta
        f_vals_new = compute_fs(rbfs, [xx, thetatheta], sig)
        new_action = epsilongreedy(Q, f_vals_new, 0, number_actions)
        Qheight[xx, thetatheta] = Qlookup(Q, f_vals_new, new_action)

x, theta = np.meshgrid(x, theta)


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x, theta, Qheight)
plt.show()

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

'''