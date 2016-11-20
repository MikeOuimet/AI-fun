from __future__ import division
from numpy.random import randint
from numpy.random import uniform
import copy
import envs.discretegame
class TicTacToe(envs.discretegame.DiscreteGame):
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



    def print_state(self, s):
        print 'state:'
        print s[0:3]
        print s[3:6]
        print s[6:9]

    def user_input(self, s):
        leg_acts = self.legal_actions(s)
        legal_flag = False
        while not legal_flag:
            user_action = raw_input('Select a location (1-9):  ')
            try:
                user_action = int(user_action)
            except:
                print 'Not an integer'
                continue
            #else:
            #    user_action = int(user_action)
            if user_action-1 in self.legal_actions(s):
                legal_flag = True
            else:
                print 'Illegal move'
        new_state, reward = self.generative_model(s, user_action - 1, 2)
        return new_state, reward


    def generative_model(self, s, a, player):
        s_prime = copy.copy(s)
        s_prime[a] = player
        reward = self.reward(s_prime)
        return s_prime, reward


    def play_TTT_MCTS(self, solver):
        game_over = False
        while not game_over:
            solver.search(self, solver.state)
            #print solver.tree
            reward = solver.update_state(self, solver.state)
            self.print_state(solver.state)
            if reward == 1.0:
                game_over = True
                print 'I won!'
                break
            elif len(self.legal_actions(solver.state)) == 0:
                game_over = True
                print 'Draw!'
                break
            next_state, reward = self.user_input(solver.state)
            solver.state = next_state
            if reward == -1.0:
                game_over = True
                print 'You won!'
                break
            elif len(self.legal_actions(solver.state)) == 0:
                game_over = True
                print 'Draw!'
                break

''' Code for main.py
import envs.tictactoe
import solvers.mcts


max_time = 1
max_depth = 10
exploration = 2
board = [0,0,0,0,0,0,0,0,0]
env = envs.tictactoe.TicTacToe(start = board)
solver = solvers.mcts.MonteCarlo(env, max_time = max_time, max_depth = max_depth, exploration= exploration)

env.play_TTT_MCTS(solver)



'''