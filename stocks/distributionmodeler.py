'''
Uses both simple mean/variance and mixture of Gaussians (using EM) modeling to
model S&P 500 annualized returns
'''

import numpy as np
import matplotlib.pyplot as plt

def GaussianProb(x, mean, var):
	return 1/(np.sqrt(2*var*np.pi))*np.exp(-(x - mean)**2/(2*var))


data = np.loadtxt("data.txt", delimiter="\t")
data_size = len(data[:,0])

data_mean = np.mean(data[:,1])
data_std = np.std(data[:,1])

samples = np.random.normal(data_mean, data_std, data_size)


n_bins = 5
'''
plt.hist(data[:,1], bins=n_bins, alpha=.5, label='True data')
plt.hist(samples, bins = n_bins, alpha=.5, label='Gaussian approx')
plt.legend(loc='upper left')
plt.show()
'''

data_mean = np.mean(data[:,1])
data_variance = np.var(data[:,1])

num_mixture = 5

mixture_mean = np.random.normal(data_mean, data_std, num_mixture)
mixture_var = data_std**2*np.ones(num_mixture)
mixture_latent_likelihood = (1.0/num_mixture)*np.ones(num_mixture)
mixture_data_likelihood = np.zeros((data_size, num_mixture))


# E-step

for iter_num in range(5000):
	for datapoint in range(data_size):
		for mix in range(num_mixture):
			mixture_data_likelihood[datapoint, mix] = GaussianProb(data[datapoint, 1], mixture_mean[mix], mixture_var[mix])
		mixture_data_likelihood[datapoint, :] = np.multiply(mixture_data_likelihood[datapoint, :], mixture_latent_likelihood) \
		/np.dot(mixture_data_likelihood[datapoint, :], mixture_latent_likelihood)


	# M-step
	mixture_latent_likelihood[:] = np.sum(mixture_data_likelihood, axis=0)/data_size
	for mix in range(num_mixture):
		mixture_mean[mix] =  np.dot(data[:,1], mixture_data_likelihood[:,mix])/np.sum(mixture_data_likelihood[:,mix])
	for mix in range(num_mixture):
		temp =0
		for datapoint in range(data_size):
			temp += mixture_data_likelihood[datapoint, mix]*(data[datapoint,1] - mixture_mean[mix])**2
		mixture_var[mix] = temp/np.sum(mixture_data_likelihood[:,mix])
	if iter_num % 500 ==0:
		print 'iteration #', iter_num
		print 'latent Gaussian probs', mixture_latent_likelihood
		print 'latent Gaussian means', mixture_mean
		print 'latent Gaussian variances', mixture_var
		print ''


mixture_samples=np.zeros(data_size)
for samp in range(data_size):
	latent_event = np.random.multinomial(1, mixture_latent_likelihood)
	latent_id = np.argmax(latent_event)
	mixture_samples[samp] = np.random.normal(mixture_mean[latent_id],np.sqrt(mixture_var[latent_id]), 1)

#plt.hist(data[:,1], bins=n_bins, alpha=.5, label='True data')
#plt.hist(samples, bins = n_bins, alpha=.5, label='Gaussian approx')
#plt.hist(mixture_samples, bins = n_bins, alpha=.5, label='Mixture approx')
#plt.legend(loc='upper left')
#plt.show()

np.savez('GMM.npz', mixture_mean=mixture_mean, mixture_var=mixture_var, mixture_latent_likelihood=mixture_latent_likelihood)