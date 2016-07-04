import numpy as np
import matplotlib.pyplot as plt
'''Basic SARSA(lambda) implementation illustrated on a die rolling problem where you get reward equal
to die roll unless roll a bad number, which makes you lose everything collected.  Actions are to
0 - keep rolling, 1- keep the money you've collected and stop playing

'''

#  Parameters
N = 10
e1 = .1
alpha = .1
gamma = 0.9
lam = 1
bad = [10]
go_on = True
nruns = 500


endstate = 1000
Q = np.zeros((endstate, 2))


#  TD(lambda)

state = 0


#  epsilon-greedy
def epsilongreedy(Qf, statef, epsilonf):
    if np.random.random() > epsilonf:
        a = Qf[statef, :].argmax()
    else:
        a = np.random.randint(0, 2)
    return a


# 1 = stop, 0 = bet

def environment(statef, a, bad):
    if a == 0:
        nature = np.random.randint(1, N+1)
        if nature in bad:
            r = -state
            goon = False
            ns = endstate-1
        else:
            r = nature
            goon= True
            ns = statef + nature
    else:
        r = 0
        goon = False
        ns = endstate-1
    return r, goon, ns


# + gamma*potential[newstate] -potential[state]
def tdlambda(Q, s, a, r, ns, e, es):
    Qnew = Q.copy()
    diff = alpha*(r + gamma*(Q[ns, :].max()) - Q[s, a])
    for ss in range(es):
        for ac in range(2):
            if e[ss, ac] > 0:
                Qnew[ss, ac] = Q[ss, ac] + diff*e[ss,ac]
    return Qnew




for run in range(nruns):
    eps = e1
    elig = np.zeros((endstate, 2))
    while go_on:
        action = epsilongreedy(Q, state, eps)
        elig[state, action] = elig[state, action] + 1
        reward, go_on, newstate = environment(state, action, bad)
        Qnew = tdlambda(Q, state, action, reward, newstate, elig, endstate)
        #if run % 10000 == 0:
        #    print run
        #    print 'The state is %r and action is %r and prior Q is (%r %r)' % (state, action, Q[state, 0], Q[state, 1])
        #    print 'The reward is %r and the new state is %r' % (reward, newstate)
        #    print 'The Q of the new state %r is (%r %r)' % (newstate, Q[newstate, 0], Q[newstate,1])
        #    print 'The updated Q of state %r is (%r %r)' % (state, Qnew[state, 0], Qnew[state, 1])
        #    print ''
        elig = gamma*lam*elig
        Q = Qnew
        state = newstate
    go_on = True
    state = 0 # np.random.randint(0, 6)

printn = 100
states = np.linspace(0, printn, printn+1)
plt.plot(states, Q[0:printn+1, 1])
plt.plot(states, Q[0:printn+1, 0])
plt.show()





