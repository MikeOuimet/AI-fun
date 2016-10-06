'''
Uses both simple mean/variance and mixture of Gaussians (using EM) modeling to
model S&P 500 annualized returns
'''

import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("data.txt", delimiter="\t")
data_size = len(data[:,0])

data_mean = np.mean(data[:,1])
data_std = np.std(data[:,1])

samples = np.random.normal(data_mean, data_std, data_size)


n_bins = 40
plt.hist(data[:,1], bins=n_bins, alpha=.5, label='True data')
plt.hist(samples, bins = n_bins, alpha=.5, label='Gaussian approx')
plt.legend(loc='upper left')
plt.show()

data_mean = np.mean(data[:,1])
data_variance = np.var(data[:,1])

num_mixture = 3

mixture_mean = np.random.normal(data_mean, data_std, num_mixture)
mixture_var = data_std**2*np.ones(num_mixture)
mixture_latent_likelihood = 1.0/num_mixture*np.ones(num_mixture)
mixture_data_likelihood = np.zeros((data_size, num_mixture))




