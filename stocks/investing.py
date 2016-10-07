import numpy as np
import matplotlib.pyplot as plt

def GMM_sample(mixture_mean, mixture_var, mixture_latent_likelihood):
	latent_event = np.random.multinomial(1, mixture_latent_likelihood)
	latent_id = np.argmax(latent_event)
	return np.random.normal(mixture_mean[latent_id],np.sqrt(mixture_var[latent_id]), 1)



npzfile = np.load('GMM.npz')

mixture_mean =npzfile['mixture_mean']
mixture_var = npzfile['mixture_var']
mixture_latent_likelihood = npzfile['mixture_latent_likelihood']


initial = 0
per_year = 1000
inflation = .03
n_runs = 10000
fraction_bonds = 0.2

years_investing = 20

amount_stocks = np.ones(shape=(n_runs, years_investing))
amount_stocks[:, 0] = initial*(1-fraction_bonds)
amount_bonds = np.ones(shape=(n_runs, years_investing))*fraction_bonds*initial

for run in range(n_runs):
	for year in range(years_investing-1):
		amount_stocks[run, year+1] = (amount_stocks[run, year]+ per_year*(1-fraction_bonds))*(1+.01*GMM_sample(mixture_mean, mixture_var, mixture_latent_likelihood) - inflation)
		amount_bonds[run, year+1] = (amount_bonds[run,year] + per_year*fraction_bonds)
		total = amount_bonds[run, year+1] + amount_stocks[run, year+1]
		amount_bonds[run, year+1] = total*fraction_bonds
		amount_stocks[run, year+1] = total*(1 -fraction_bonds)
amount = amount_bonds + amount_stocks
mean_of_runs = np.mean(amount, axis = 0)
min_of_runs = np.min(amount, axis = 0)
max_of_runs = np.max(amount, axis = 0)

plt.plot(mean_of_runs)
plt.plot(min_of_runs)
plt.plot(max_of_runs)
plt.show()
#for i in range(n_runs):
#	plt.plot(amount[i,:])
#plt.show()