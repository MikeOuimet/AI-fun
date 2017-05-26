import math
import numpy as np


def signed_angle(a1, a2):
	'''calculates the signed difference between 2 angles
	positive, counter-clockwise, negative, clockwise '''
	sign = [1., -1.]
	dcc = (a2- a1) % (2*math.pi)
	dc = (a1 - a2) % (2*math.pi)
	d = min(dc, dcc)
	arg = np.argmin([dc, dcc])
	signed_d = sign[arg]*d
	return signed_d


if __name__ == '__main__':
	a1 = math.pi/4
	a2 = -3.

	print signed_angle(a1, a2)

	print signed_angle(a2, a1)