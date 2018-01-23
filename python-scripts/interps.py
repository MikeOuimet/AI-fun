import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata
import time

global precision 
precision = 0.1


def func(x, y):
	return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

def my_interp(points, vals):
	numpoints = len(vals)
	A = np.empty(shape = (numpoints,4))
	for idx, point in enumerate(points):
		A[idx, 0] = 1
		A[idx, 1] = point[0]
		A[idx, 2] = point[1]
		A[idx, 3] = point[0]*point[1]

	inv_m = np.linalg.inv(np.dot(A.T, A))

	pseudo = np.dot(inv_m, A.T)
	coeffs = np.dot(pseudo, vals)
	return coeffs



class InterpSet:
	def __init__(self, pt, val):
		self.neighs = np.array([pt])
		self.vals = np.array([[val]])

	def add_point(self, pt, val):
		self.neighs = np.concatenate((self.neighs, np.array([list(pt)])), axis = 0)
		self.vals = np.concatenate((self.vals, np.array([[val]])), axis = 0)


def add_point(d, hashpoint, point, data_point):
	if hashpoint not in d.keys():
		d[hashpoint] = InterpSet(point, data_point)
	else:
		d[hashpoint].add_point(point, data_point)

	return d

def my_hash(point):
	return tuple(np.round(point/precision)*precision)

def get_data(d, point):
	hashpoint = my_hash(point)
	return d[hashpoint].neighs, d[hashpoint].vals


d = {}

points = np.random.uniform(0, 1, size = (10000, 2))

data = func(points[:, 0], points[:, 1])



for point, data_point in zip(points, data):
	hashpoint = my_hash(point)
	d = add_point(d, hashpoint, point, data_point )

my_points, my_vals = get_data(d, np.array([0.2, 0.78]))
my_interp(my_points, my_vals)


testset = np.random.uniform(0, 1, size = (10, 2))
interp1 = LinearNDInterpolator(points, data)

s = time.time()
ans = interp1(testset)
lt = time.time() - s
print "LinearInterp", lt

s = time.time()
interp2 = griddata(points, data, testset, method = 'nearest')
print "Grid Interp", time.time() - s


s = time.time()
for tp in testset:
	points, data = get_data(d, tp)
	#print len(points), len(data)
	#my_interp(points, data)
	#print points, data
	#time.sleep(10000)
	lm = time.time() - s
print "My hash", lm


print "My hash is {}x faster".format(lt/lm)

