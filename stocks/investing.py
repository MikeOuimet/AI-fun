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


initial = 100
per_year = 46
inflation = .025
n_runs = 8000
fraction_bonds = 0.2

years_investing = 40

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
sd_of_runs = np.std(amount,axis = 0)
plus_2sd = mean_of_runs + 2*sd_of_runs
minus_2sd = mean_of_runs - 2*sd_of_runs
plus_1sd = mean_of_runs + 1*sd_of_runs
minus_1sd = mean_of_runs - 1*sd_of_runs

'''
min_year = np.zeros(n_runs)
for run in range(n_runs):
	min_year[run] = np.min([i for i,v in enumerate(amount[run,:]) if v > 1500])

print np.mean(min_year)
print np.std(min_year)
'''


#plt.plot(max_of_runs, label ='max')
#plt.plot(plus_2sd, label ='+2sd')
plt.plot(plus_1sd, label ='+1sd')
plt.plot(mean_of_runs, label='mean')
plt.plot(minus_1sd, label ='-1sd')
#plt.plot(minus_2sd, label ='-2sd')
#plt.plot(min_of_runs, label ='min')
plt.legend(loc='upper left')
plt.show()
#for i in range(n_runs):
#	plt.plot(amount[i,:])
#plt.show()


plt.hist(amount[:,11], bins=100, alpha=1)
plt.show()

