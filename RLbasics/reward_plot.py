import numpy as np
import matplotlib.pyplot as plt

data = np.load('rewards.npy')
mean_data = np.zeros(len(data))
window = 200
for i in range(len(data)):
	if i > window:
		mean_data[i] =np.sum(data[i-window:i])/window
	else:
		mean_data[i] =np.sum(data[0:i])/(i+1.0)

plt.plot(mean_data)
plt.show()
