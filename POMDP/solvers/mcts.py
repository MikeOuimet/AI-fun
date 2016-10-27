import datetime

class MonteCarlo(object):
    # fully observable UCT so state is sufficient
    def __init__(self, env, **kwargs):
        seconds = kwargs.get('max_time', 1)
        self.gamma = kwargs.get('gamma', 1.0)
        self.max_depth =kwargs.get('max_depth', 20)
        self.calculation_time = datetime.timedelta(seconds=seconds)
        self.start = env.start
        self.state = self.start
        self.visit_init = 1
        self.value_init = 0
        self.visits = {tuple(self.start): self.visit_init}
        self.values =  {tuple(self.start): self.value_init}
        #self.history = env.h

    def simulate(self, env, s, depth):
        if depth > self.max_depth:
            return 0
        if tuple(s) not in self.visits:
            for move in env.legal_actions(s):
                new_state, reward = env.generative_model(s, move, 1)
                self.visits[tuple(new_state)] = self.visit_init
                self.values[tuple(new_state)] = reward
            return self.rollout(env, s, depth)
        pass


    def rollout(self, env, s, depth):
        if depth > self.max_depth:
            return 0
        a = env.rollout_policy(s) #random?
        if a == 'Draw':
            return 0
        new_state, reward, done = env.two_ply_generative(s, a)
        print new_state
        if done:
            return reward
        else:
            return reward + self.gamma*self.rollout(env, new_state, depth+1)

    def search(self, s):
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            # s ~ B(h)
            # self.simulate(s, 0)
            pass
        print '{} seconds have elapsed'.format(self.calculation_time)