from __future__ import division
import numpy as np
from numpy.random import randint
from numpy.random import uniform
import copy

class DiscreteGame(object):


    def test(self, s):
       print s


    def two_ply_generative(self, solver, s, a):
        player = 1
        done = False
        s_prime, reward = self.generative_model(s, a, player)
        if reward == 1.0: # won
            done = True
            return s_prime, reward, done
        if len(self.legal_actions(s_prime)) == 0:
            done = True
            return s_prime, 0.0, done
        else:
            player = 2
            opponent_action = self.opponent_rollout(solver, s_prime)
            s_prime, reward = self.generative_model(s_prime, opponent_action, player)
            if reward == -1.0:
                done = True
            return s_prime, reward, done


    def legal_next_states(self, s, player):
        acts = self.legal_actions(s)
        next_states = []
        for act in acts:
            s_prime, reward = self.generative_model(s, act, player)
            next_states.append(s_prime)
        return next_states


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
            for act in acts:
                s_prime, reward = self.generative_model(s, act, 1)
                if reward == 1.0:
                    return act
            return acts[randint(0, len(acts))]
        else:
            return 'Draw'

    def play_game_MCTS(self, solver):
        game_over = False
        self.print_state(solver.state)
        if self.human_first:
            next_state, reward = self.user_input(solver.state)
            solver.state = next_state
        while not game_over:
            solver.search(self, solver.state)
            #print solver.tree
            reward = solver.update_state(self, solver.state)
            if reward == 'Draw':
                print 'Draw!'
                break
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


    def user_input(self, s):
        leg_acts = self.legal_actions(s)
        legal_flag = False
        while not legal_flag:
            user_action = raw_input(self.user_string)
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