import numpy as np

def example(n):
	i= 0
	data = np.empty(shape = (1, n))
	while True:
		for i in range(n):
			data[0, i] = i
		else:
			break