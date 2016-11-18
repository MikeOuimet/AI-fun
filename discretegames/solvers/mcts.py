from __future__ import division
import datetime
from math import sqrt, log
import numpy as np



class MonteCarlo(object):
    # fully observable UCT so state is sufficient
    def __init__(self, env, **kwargs):
        seconds = kwargs.get('max_time', 1)
        self.gamma = kwargs.get('gamma', 1.0)
        self.max_depth =kwargs.get('max_depth', 20)
        self.c = kwargs.get('exploration', 2)
        self.calculation_time = datetime.timedelta(seconds=seconds)
        self.start = env.start
        self.state = self.start
        self.player = 1
        self.visit_init = 1
        self.value_init = 0.0
        self.tree = {}
        #self.add_tree_layer(env, self.start)


    def add_tree_layer(self, env, s):
        #print self.to_tuple(s)
        if self.to_tuple(s) not in self.tree:
            self.tree[self.to_tuple(s)] = [self.value_init, self.visit_init]
            for move in env.legal_actions(s):
                new_state, reward = env.generative_model(s, move, self.player)
                self.tree[self.to_tuple(new_state)] = [reward, self.visit_init]


    def simulate(self, env, s, depth):
        if depth > self.max_depth:
            return 0
        if self.to_tuple(s) not in self.tree:
            self.add_tree_layer(env, s)
            return self.rollout(env, s, depth)      
        new_a = self.UCT_sample(env, s)
        if new_a == 'done':  # Should this be in the environment?
            return 0.0        #
        s_prime, reward, done = env.two_ply_generative(self, s, new_a)
        if done:
            total_reward =  reward
        else:
            total_reward = reward + self.gamma*self.simulate(env, s_prime, depth+1)
        state_my_action, reward = env.generative_model(s, new_a, self.player)
        self.tree[self.to_tuple(s)][1] += 1
        self.tree[self.to_tuple(s)][0] += (total_reward - self.tree[self.to_tuple(s)][0])\
        /(self.tree[self.to_tuple(s)][1])  # added so opponent can use
        self.tree[self.to_tuple(state_my_action)][1] += 1
        self.tree[self.to_tuple(state_my_action)][0] += (total_reward - self.tree[self.to_tuple(state_my_action)][0])\
        /(self.tree[self.to_tuple(state_my_action)][1])
        #if done:
        #
        return total_reward


    def UCT_sample(self, env, s):
        acts = env.legal_actions(s)
        if len(acts) ==0:       # Should this be in the environment?
            return 'done'       #
        act_choice = acts[0] # use first action as initial guess 
        guess_next_state, reward = env.generative_model(s, act_choice, self.player)
        key = guess_next_state
        value = self.tree[self.to_tuple(key)][0]
        for act in acts:
            state, r = env.generative_model(s, act, self.player)
            val = self.tree[self.to_tuple(state)][0] + \
            self.c*sqrt(log(self.tree[self.to_tuple(s)][1]))/(sqrt(self.tree[self.to_tuple(state)][1]))
            if val > value:
                value = val
                act_choice = act
        return act_choice


    def best_action(self, env, s):
        acts = env.legal_actions(s)
        act_choice = acts[0] # use first action as initial guess 
        guess_next_state, reward = env.generative_model(s, act_choice, self.player)
        key = guess_next_state
        value = self.tree[self.to_tuple(key)][0]
        for act in acts:
            state, r = env.generative_model(s, act, self.player)
            val = self.tree[self.to_tuple(state)][0] 
            if val > value:
                value = val
                act_choice = act
        return act_choice


    def rollout(self, env, s, depth):
        if depth > self.max_depth:
            return 0
        a = env.rollout_policy(s) #random?
        if a == 'Draw':
            return 0.0
        new_state, reward, done = env.two_ply_generative(self, s, a)
        #print new_state
        if done:
            return reward
        else:
            return reward + self.gamma*self.rollout(env, new_state, depth+1)


    def search(self, env, s):
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.simulate(env, s, 0)
        print '{} seconds have elapsed'.format(self.calculation_time)
        self.stats_next_states(env, s)
        
    def update_state(self, env, s):
        next_a = self.best_action(env, s)
        new_state, reward = env.generative_model(s, next_a, self.player)
        self.state = new_state
        return reward


    def stats_next_states(self, env, s):
        next_states = env.legal_next_states(s, self.player)
        for state in next_states:
            #print state
            env.print_state(state)
            print 'value', self.tree[self.to_tuple(state)][0]
            print 'times visited', self.tree[self.to_tuple(state)][1]
            print 'upper confidence bound', self.tree[self.to_tuple(state)][0] + self.c*sqrt(log(self.tree[self.to_tuple(s)][1]))/(sqrt(self.tree[tuple(state)][1]))
            print ''

    def to_tuple(self, s):
        if type(s) == list:
            return tuple(s)
        elif type(s) == np.ndarray:   # Figure out how to check for ndarray type
            flat = s.flatten()
            return tuple(flat)