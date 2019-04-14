
def mnist_digits_distance():

    import numpy as np
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from tatter import two_sample_test
    from sklearn.metrics import pairwise_distances
    import os

    plot_path = './tatter/tests/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    digits = datasets.load_digits()
    data = digits.data
    target = digits.target

    num = [data[target == i] for i in range(10)]

    plt.figure(figsize=(16, 16))

    for i in range(1, 9):
        for j in range(8):
            ax = plt.subplot(9, 8, i * 8 + j + 1)

            if i == 1: ax.set_title('%i' % (j + 1), size=25, fontweight="bold")
            if j == 0: ax.set_ylabel('%i' % (i + 1), size=25, fontweight="bold")
            ax.set_yticklabels([])

            if i != 8: ax.set_xticklabels([])

            if i <= j:
                plt.axis('off')
                continue

            sigma2 = np.median(pairwise_distances(num[i + 1], num[j + 1], metric='euclidean')) ** 2 * 2.0
            mmd2u, mmd2u_null, p_value = two_sample_test(num[i+1], num[j+1], model='MMD',
                                                 kernel_function='rbf', gamma=1.0/sigma2,
                                                 iterations=500, verbose=True, n_jobs=4)

            # print mmd2u
            prob, bins, patches = plt.hist(mmd2u_null, range=[-0.005, 0.1], bins=5, normed=True)
            plt.plot(mmd2u, prob.max() / 10, 'wv', markersize=14, markeredgecolor='k',
                     markeredgewidth=2, label="$MMD^2_u$   p-value < %0.3f" % p_value)
            plt.xlim([-0.01, 0.5])
            if i == 8 and j == 3:
                ax.set_xlabel('$MMD^2_u$ estimate', size=25)
            # plt.legend(loc=1, numpoints=1, prop={'size':15})

    plt.savefig(plot_path + 'MNIST-test.pdf', bbox_inches='tight')


    plt.figure(figsize=(24, 2))

    for i in range(9):

        ax = plt.subplot(1, 9, i+1)

        ax.set_title('%i' % (i + 1), size=25, fontweight="bold")
        if i == 0: ax.set_ylabel('PDF', size=25)
        ax.set_yticklabels([])

        sigma2 = np.median(pairwise_distances(num[i + 1][::2], num[i + 1][1::2], metric='euclidean')) ** 2 * 2.0
        mmd2u, mmd2u_null, p_value = two_sample_test(num[i+1][::2], num[i+1][1::2], model='MMD',
                                                 kernel_function='rbf', gamma=1.0/sigma2,
                                                 iterations=5000, verbose=True, n_jobs=4)

        # print mmd2u
        prob, bins, patches = plt.hist(mmd2u_null, range=[-0.005, 0.01], bins=50, normed=True)
        plt.plot(mmd2u, prob.max() / 10, 'wv', markersize=14, markeredgecolor='k',
                     markeredgewidth=2, label="$MMD^2_u$   p-value < %0.3f" % p_value)
        plt.xlim([-0.005, 0.01])
        if i == 4:
            ax.set_xlabel('$MMD^2_u$ estimate', size=25)
        # plt.legend(loc=1, numpoints=1, prop={'size':15})
        ax.set_xticks([-0.005, 0, 0.005, 0.01 ])
        ax.set_xticklabels([-0.005, '0', 0.005, ' ' ], size=14)

    plt.savefig(plot_path + 'MNIST-test-2.pdf', bbox_inches='tight')