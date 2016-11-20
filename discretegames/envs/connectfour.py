from __future__ import division
import numpy as np
from numpy.random import randint
from numpy.random import uniform
import copy
import envs.discretegame


class ConnectFour(envs.discretegame.DiscreteGame):
    def __init__(self, **kwargs):
        self.start = kwargs.get('start', np.zeros(shape=(6,7), dtype = np.int))
        self.state = self.start
        self.player = 1


    def legal_actions(self, s):
        # s = (rows,columns)
        acts = range(7)
        legal_acts = []
        last_row = s[5,:]
        for col in range(7):
            if last_row[col]== 0:
                legal_acts.append(acts[col])
        return legal_acts


    def generative_model(self, s, a, player):
        loc = 5
        while s[loc, a] == 0 and s[loc-1, a] ==0 and loc > 0:
            loc -= 1
        s_prime = copy.copy(s)
        s_prime[loc, a] = player
        reward = self.reward(s_prime)
        return s_prime, reward


    def print_state(self, s):
        for row in range(6):
            print s[5-row, :]


    def win_in_row(self, v):
        if len(v) >=4:
            for spot in range(len(v)- 3):
                if v[spot] != 0:
                    if v[spot]== v[spot+1] and v[spot + 1] == v[spot+2] and v[spot + 2] == v[spot+3]:
                        return v[spot]
        return 0


    def reward(self, s):
        for col in range(7):
            vec = s[:, col]
            check = self.win_in_row(vec)
            if check != 0:
                if check ==1:
                    return 1.0
                else:
                    return -1.0
        for row in range(6):
            vec = s[row, :]
            check = self.win_in_row(vec)
            if check != 0:
                if check ==1:
                    return 1.0
                else:
                    return -1.0
        diag_check = self.check_diags(s)
        if diag_check != 0:
            if check ==1:
                return 1.0
            else:
                return -1.0
        diag_check = self.check_diags(s[::-1])
        if diag_check != 0:
            if check ==1:
                return 1.0
            else:
                return -1.0
        return 0.0


    def check_diags(self, s):
        for row in range(6):
            diag =[s[row, 0]]
            row_place =row
            col_place = 0
            while col_place < 6 and row_place < 5:
                diag.append(s[row_place + 1, col_place + 1])
                row_place += 1
                col_place += 1
            check = self.win_in_row(diag)
            if check != 0:
                return check
        for col in range(1, 7):
            diag =[s[0, col]]
            row_place =0
            col_place = col
            while col_place < 6 and row_place < 5:
                diag.append(s[row_place + 1, col_place + 1])
                row_place += 1
                col_place += 1
            check = self.win_in_row(diag)
            if check != 0:
                return check
        return 0.0

