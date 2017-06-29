import dubins
import matplotlib.pyplot as plt
import numpy as np

q0 = (0., 0., 0.)
q1 = (10, 5, -np.pi)
turning_radius = 1.0
step_size = 0.25

qs, _ = dubins.path_sample(q0, q1, turning_radius, step_size)

data = np.array(qs)

#print data

plt.scatter(data[:,0], data[:,1])
plt.show()