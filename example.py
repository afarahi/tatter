
from tatter import two_sample_test

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

np.random.seed(100)

n = 200

# draw random points from a log-normal and a gaussian distributions
Y = np.random.lognormal(mean=2, sigma=0.3, size=n)
X = np.random.normal(loc=np.exp(2.0+0.3**2/2.0), scale=0.3*np.exp(2.0), size=n)

XX = X[:, np.newaxis]
YY = Y[:, np.newaxis]

# compute MMD/K-S two sample tests and their null distributions
sigma2 = np.median(pairwise_distances(XX, YY, metric='euclidean'))**2 * 2.0
mmd2u, mmd2u_null, p_value = two_sample_test(XX, YY, model='MMD',
                                             kernel_function='rbf', gamma=1.0/sigma2,
                                             iterations=5000, verbose=True, n_jobs=1)

ks, ks_null, ks_p_value = two_sample_test(XX, YY, model='KS', iterations=5000,
                                          verbose=True, n_jobs=1)

# visualize the results
plt.figure(figsize=(6, 9))
plt.subplot(211)
prob, bins, patches = plt.hist(mmd2u_null, range=[-0.005, 0.015], bins=50,
                               density=True, color='green', label='Null distribution')
plt.plot(mmd2u, prob.max()/25, 'wv', markersize=14, markeredgecolor='k',
         markeredgewidth=2, label="$MMD^2_u$   p-value = %0.3f"%p_value)
plt.xlim([-0.005, 0.020])
plt.ylabel('PDF', size=20)
plt.legend(loc=1, numpoints=1, prop={'size':15})

plt.subplot(212)
prob, bins, patches = plt.hist(ks_null, range=(0.0, 0.25), bins=50,
                               density=True, color='orange', label='Null distribution')
plt.plot(ks, prob.max()/25, 'wv', markersize=14, markeredgecolor='k',
         markeredgewidth=2, label="$K-S$   p-value = %0.3f"%ks_p_value)
plt.xlim([0.0, 0.25])
plt.ylabel('PDF', size=20)
plt.legend(loc=1, numpoints=1, prop={'size':15})

plt.show()
