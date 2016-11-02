import numpy as np
class ConnectFour(object):
	def __init__(self, **kwargs):
		self.start = kwargs.get('start', np.zeros(shape=(6,7), dtype = np.int))
