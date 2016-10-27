from numpy.random import randint
import copy
class TicTacToe(object):
    def __init__(self, **kwargs):
        self.start = kwargs.get('start', [0,0,0,0,0,0,0,0,0])
        self.s = self.start
        #self.h= ''

    def legal_actions(self, s):
        return [i for i,v in enumerate(s) if v == 0]

    def reward(self, s):
        for i in [0, 3, 6]:
            if s[i]==s[i+1] and s[i+1] == s[i+2] and s[i] != 0:
                if s[i]==1:
                    return 1.0
                else:
                    return -1.0
        for i in [0, 1, 2]:
            if s[i]==s[i+3] and s[i+3]==s[i+6] and s[i] != 0:
                if s[i] ==1:
                    return 1.0
                else:
                    return -1.0
        if s[0]==s[4] and s[4] ==s[8] and s[0] != 0:
            if s[0]==1:
                return 1.0
            else:
                return -1.0
        if s[2]==s[4] and s[4]==s[6] and s[2] != 0:
            if s[2]==1:
                return 1.0
            else:
                return -1.0
        else:
            return 0.0

    def generative_model(self, s,a, player):
        s_prime = copy.copy(s)
        s_prime[a] = player
        reward = self.reward(s_prime)
        return s_prime, reward 


    def two_ply_generative(self, s, a):
        player = 1
        s_prime, reward = self.generative_model(s, a, player)
        done = False
        if reward == 1.0: # won
            done = True
            return s_prime, reward, done 
        else:
            player = 2
            opponent_action = self.rollout_policy(s_prime)
            s_prime, reward = self.generative_model(s_prime, opponent_action, player)
            if reward == -1.0:
                done = True
            return s_prime, reward, done

    def rollout_policy(self, s):
        '''chooses actions for rollout policy'''
        acts = self.legal_actions(s)
        if len(acts) >0:
            return acts[randint(0, len(acts))]
        else:
            return 'Draw'


