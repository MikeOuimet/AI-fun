from __future__ import division
import numpy as np
from numpy.random import randint
from numpy.random import uniform
import copy

class DiscreteGame(object):
    def test(self, s):
       print s

    def generative_model(self, s,a, player):
        s_prime = copy.copy(s)
        s_prime[a] = player
        reward = self.reward(s_prime)
        return s_prime, reward

    def two_ply_generative(self, solver, s, a):
        player = 1
        done = False
        s_prime, reward = self.generative_model(s, a, player)
        if reward == 1.0: # won
            done = True
            return s_prime, reward, done
        if len(self.legal_actions(s_prime)) ==0:
            done = True
            return s_prime, 0.0, done
        else:
            player = 2
            opponent_action = self.opponent_rollout(solver, s_prime)
            s_prime, reward = self.generative_model(s_prime, opponent_action, player)
            if reward == -1.0:
                done = True
            return s_prime, reward, done


    def opponent_rollout(self,solver, s):
        acts = self.legal_actions(s)
        if len(acts) > 0:
            some_in_tree = False
            best_in_tree = 100
            best_action = 100
            count = 0
            for act in acts:
                s_prime, reward = self.generative_model(s, act, 2)
                if reward == -1.0:
                    return act
                if solver.to_tuple(s_prime) in solver.tree:
                    count += 1
                    if solver.tree[solver.to_tuple(s_prime)][0] < best_in_tree:
                        best_in_tree = solver.tree[solver.to_tuple(s_prime)][0]
                        best_action = act
            if count > 0:
                if uniform(0,1) > count/len(acts):
                    return best_action
                else:
                    return acts[randint(0, len(acts))]
            else:
                return acts[randint(0, len(acts))]
        else:
            return 'Draw'

    def rollout_policy(self, s):
        '''chooses actions for rollout policy'''
        acts = self.legal_actions(s)
        if len(acts) >0:
            #for action in acts:
            #    ns, reward = generative_model(s, action,)
            #
            return acts[randint(0, len(acts))]
        else:
            return 'Draw'