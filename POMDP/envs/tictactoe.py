from numpy.random import randint
import copy
class TicTacToe(object):
    def __init__(self, **kwargs):
        self.start = kwargs.get('start', [0,0,0,0,0,0,0,0,0])
        self.s = self.start
        #self.h= ''


    def legal_actions(self, s):
        return [i for i,v in enumerate(s) if v == 0]


    def legal_next_states(self, s, player):
        acts = self.legal_actions(s)
        next_states = []
        for act in acts:
            s_prime, reward = self.generative_model(s, act, player)
            next_states.append(s_prime)
        return next_states


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
        #print 'The player is', player
        #print a
        #if a == 'Draw'
        #    return s, 0.0
        s_prime[a] = player
        reward = self.reward(s_prime)
        #done = False
        #if len(self.legal_actions(s_prime)) == 0:
        #    done = True
        return s_prime, reward


    def two_ply_generative(self, s, a):
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
            opponent_action = self.opponent_rollout(s_prime)
            s_prime, reward = self.generative_model(s_prime, opponent_action, player)
            if reward == -1.0:
                done = True
            return s_prime, reward, done


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

    def opponent_rollout(self, s):
        acts = self.legal_actions(s)
        if len(acts) > 0:
            for act in acts:
                s_prime, reward = self.generative_model(s, act, 2)
                if reward == -1.0:
                    return act
            return acts[randint(0, len(acts))]
        else:
            return 'Draw'



    def print_state(self, s):
        print 'state:'
        print s[0:3]
        print s[3:6]
        print s[6:9]

    def user_input(self, s):
        user_action = input('Select a location:')
        #if user_action not in self.legal_actions(s):
        #    self.user_input(s)
        new_state, reward = self.generative_model(s, user_action, 2)
        return new_state, reward

    def play_TTT_MCTS(self, solver, ):
        game_over = False
        while not game_over:
            solver.search(self, solver.state)
            #print solver.tree[tuple([2,0,1,0,2,2,0,1,1])]
            #print solver.tree[tuple([2,0,1,2,2,2,0,1,1])]
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

