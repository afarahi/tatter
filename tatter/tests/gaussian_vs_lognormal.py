def test_gaussian_vs_lognormal():

    from tatter import two_sample_test
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import pairwise_distances
    import os

    plot_path = './tatter/tests/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    np.random.seed(100)

    n = 200

    # draw random points from a log-normal and a gaussian distributions
    Y = np.random.lognormal(mean=2, sigma=0.3, size=n)
    X = np.random.normal(loc=np.exp(2.0+0.3**2/2.0), scale=0.3*np.exp(2.0), size=n)

    XX = X[:, np.newaxis]
    YY = Y[:, np.newaxis]

    # compute MMD/KS/KL two sample tests and their null distributions
    sigma2 = np.median(pairwise_distances(XX, YY, metric='euclidean'))**2 * 2.0
    mmd2u, mmd2u_null, p_value = two_sample_test(XX, YY, model='MMD', kernel_function='rbf', gamma=1.0/sigma2,
                                                 iterations=1000, verbose=True, n_jobs=1)

    ks, ks_null, ks_p_value = two_sample_test(XX, YY, model='KS', iterations=1000, verbose=True, n_jobs=1)
    kl, kl_null, kl_p_value = two_sample_test(XX, YY, model='KL', iterations=1000, verbose=True, n_jobs=1)

    # visualize the results
    plt.figure(figsize=(6, 9))
    plt.subplot(311)
    prob, bins, patches = plt.hist(mmd2u_null, range=[-0.005, 0.015], bins=50, normed=True, color='green')
    plt.plot(mmd2u, prob.max()/25, 'wv', markersize=14, markeredgecolor='k',
             markeredgewidth=2, label="$MMD^2_u$   p-value = %0.3f"%p_value)
    plt.xlim([-0.005, 0.020])
    plt.ylabel('PDF', size=20)
    plt.legend(loc=1, numpoints=1, prop={'size':15})

    plt.subplot(312)
    prob, bins, patches = plt.hist(ks_null, range=(0.0, 0.25), bins=50, normed=True, color='orange')
    plt.plot(ks, prob.max()/25, 'wv', markersize=14, markeredgecolor='k',
             markeredgewidth=2, label="$K-S$   p-value = %0.3f"%ks_p_value)
    plt.xlim([0.0, 0.25])
    plt.ylabel('PDF', size=20)
    plt.legend(loc=1, numpoints=1, prop={'size':15})

    plt.subplot(313)
    prob, bins, patches = plt.hist(kl_null, range=(-0.5, 0.7), bins=50, normed=True, color='steelblue')
    plt.plot(kl, prob.max()/25, 'wv', markersize=14, markeredgecolor='k',
             markeredgewidth=2, label="$K-L$    p-value = %0.3f"%kl_p_value)
    plt.xlim([-0.5, 0.7])
    plt.xlabel('Null distribution', size=20)
    plt.ylabel('PDF', size=20)
    plt.legend(loc=1, numpoints=1, prop={'size':15})

    plt.savefig(plot_path + 'log-normal-vs-normal-p-val.pdf', bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(6, 5))
    prob, bins, patches = plt.hist(X, range=(0.0, 20), bins=20, alpha=0.5,
                                   color='steelblue', normed=True, label='Normal')
    prob, bins, patches = plt.hist(Y, range=(0.0, 20), bins=20, alpha=0.5,
                                   color='lightcoral', normed=True, label='log-Normal')
    plt.legend(loc=1, numpoints=1, prop={'size':16})

    plt.xlabel(r'$x \in \mathcal{X}$', size=18)
    plt.ylabel('PDF', size=18)

    plt.savefig(plot_path + 'log-normal-vs-normal-dist.pdf', bbox_inches='tight')
    # plt.show()
