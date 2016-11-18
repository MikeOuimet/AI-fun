from __future__ import division
import numpy as np
from numpy.random import randint
from numpy.random import uniform
import copy
import envs.discretegame

#class DiscreteGame:
#	def test(self, s):
#		print s

class ConnectFour(envs.discretegame.DiscreteGame):
	def __init__(self, **kwargs):
		self.start = kwargs.get('start', np.zeros(shape=(6,7), dtype = np.int))


	def legal_actions():
		pass


	def legal_next_states(self, s, player):
		pass


	def reward(self, s):
		pass


