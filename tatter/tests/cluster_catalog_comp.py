
def if_RM_clusters_exists():

    from pathlib import Path
    import os

    data_path = "./tatter/tests/data/"

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # if cluster catalogs does noe exists download them
    for fname in ["redmapper_sva1_public_v6.3_catalog.fits", "redmapper_dr8_public_v6.3_catalog.fits"]:

        fname_file = Path(data_path + fname)

        if not fname_file.is_file():
            import wget, gzip, shutil
            url = 'http://risa.stanford.edu/redmapper/v6.3/%s.gz' % fname
            _ = wget.download(url, out=data_path[2:])
            with gzip.open(data_path + fname + '.gz', 'rb') as f_in:
                with open(data_path + fname, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(data_path + fname + '.gz')


def RM_clusters_witness_function():

    import numpy as np
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from sklearn.metrics import pairwise_distances
    from tatter import two_sample_test, test_statistics, witness_function
    import os

    data_path = "./tatter/tests/data/"
    plot_path = './tatter/tests/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # if cluster catalogs does noe exists download them
    if_RM_clusters_exists()

    plt.figure(figsize=(12, 12))

    for i in range(4):

        zmin = 0.2 + 0.05*i
        zmax = 0.2 + 0.05*(i+1)

        # load the datasets
        RM_SDSS = fits.open(data_path+'redmapper_dr8_public_v6.3_catalog.fits')[1].data
        RM_SV = fits.open(data_path+'redmapper_sva1_public_v6.3_catalog.fits')[1].data

        # apply a redshift and richness cut
        RM_SDSS = RM_SDSS[RM_SDSS.Z_LAMBDA >= zmin]
        RM_SDSS = RM_SDSS[RM_SDSS.Z_LAMBDA < zmax]
        RM_SDSS = RM_SDSS[RM_SDSS.LAMBDA > 20]

        RM_SV = RM_SV[RM_SV.Z_LAMBDA >= zmin]
        RM_SV = RM_SV[RM_SV.Z_LAMBDA < zmax]
        RM_SV = RM_SV[RM_SV.LAMBDA > 20]

        # map richness into a log space
        RM_SDSS.LAMBDA = np.log(RM_SDSS.LAMBDA)
        RM_SV.LAMBDA = np.log(RM_SV.LAMBDA)

        # make a numpy array
        SDSS = np.array(RM_SDSS.LAMBDA)[:, np.newaxis]
        SV = np.array(RM_SV.LAMBDA)[:, np.newaxis]

        # compute the hyper-parameter
        sigma2 = np.median(pairwise_distances(SDSS, SV, metric='euclidean')) ** 2 * 2.0

        # estimate the null distribution and p-value of MMD^2_u test.
        grid = np.linspace(np.log(20), np.log(100), 1001)[:, np.newaxis]
        wf = witness_function(SV, SDSS, grid, kernel_function='rbf', gamma=1.0 / sigma2)

        ax = plt.subplot(2, 2, i+1)
        plt.plot(np.exp(grid), wf, color='black', lw=2.0)

        plt.xlim([20, 100])
        plt.title('$ %0.2f \leq z < %0.2f$'%(zmin, zmax), size=14)

        if i%2 == 0: ax.set_ylabel('witness function', size=22)
        if i > 1: ax.set_xlabel(r'$\lambda_{\rm RM}$', size=22)

    plt.savefig(plot_path + 'SDSS-vs-SV-witness-function.pdf', bbox_inches='tight')


def RM_clusters_consistency_check():

    import numpy as np
    from astropy.io import fits
    from tatter import two_sample_test
    import matplotlib.pyplot as plt
    from sklearn.metrics import pairwise_distances
    import os

    plt.figure(figsize=(12, 12))

    data_path = "./tatter/tests/data/"
    plot_path = './tatter/tests/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # if cluster catalogs does noe exists download them
    if_RM_clusters_exists()

    for i in range(4):

        zmin = 0.2 + 0.05*i
        zmax = 0.2 + 0.05*(i+1)

        # load the datasets
        RM_SDSS = fits.open(data_path+'redmapper_dr8_public_v6.3_catalog.fits')[1].data
        RM_SV = fits.open(data_path+'redmapper_sva1_public_v6.3_catalog.fits')[1].data

        # apply a redshift cut
        RM_SDSS = RM_SDSS[RM_SDSS.Z_LAMBDA >= zmin]
        RM_SDSS = RM_SDSS[RM_SDSS.Z_LAMBDA < zmax]

        RM_SV = RM_SV[RM_SV.Z_LAMBDA >= zmin]
        RM_SV = RM_SV[RM_SV.Z_LAMBDA < zmax]

        # map richness into a log space
        RM_SDSS.LAMBDA = np.log(RM_SDSS.LAMBDA)
        RM_SV.LAMBDA = np.log(RM_SV.LAMBDA)

        # normalize richness -- (x - x.median) / x.std
        std = np.std(RM_SDSS.LAMBDA); med = np.median(RM_SDSS.LAMBDA)

        RM_SV.LAMBDA = (RM_SV.LAMBDA - med) / std
        RM_SDSS.LAMBDA = (RM_SDSS.LAMBDA - med) / std

        # normalize redshift -- (x - x.median) / x.std
        std = np.std(RM_SDSS.Z_LAMBDA); med = np.median(RM_SDSS.Z_LAMBDA)

        RM_SV.Z_LAMBDA = (RM_SV.Z_LAMBDA - med) / std
        RM_SDSS.Z_LAMBDA = (RM_SDSS.Z_LAMBDA - med) / std

        # make a numpy array
        SDSS = np.array([RM_SDSS.LAMBDA, RM_SDSS.Z_LAMBDA]).T
        SV = np.array([RM_SV.LAMBDA, RM_SV.Z_LAMBDA]).T

        # compute the hyper-parameter
        sigma2 = np.median(pairwise_distances(SDSS, SV, metric='euclidean')) ** 2 * 2.0
        # estimate the null distribution and p-value of MMD^2_u test.
        mmd2u, mmd2u_null, p_value = two_sample_test(SDSS, SV, model='MMD',
                                                     kernel_function='rbf', gamma=1.0/sigma2,
                                                     iterations=500, verbose=True, n_jobs=4)

        ax = plt.subplot(2, 2, i+1)
        prob, bins, patches = ax.hist(mmd2u_null, range=(-0.01, 0.03), bins=100, normed=True)
        if p_value <= 0.001:
            plt.plot(mmd2u, prob.max() / 25, 'wv', markersize=22, markeredgecolor='k',
                     markeredgewidth=2, label='$MMD^2_u$ ($p$-value < %0.3f)'%p_value)
        else:
            plt.plot(mmd2u, prob.max() / 25, 'wv', markersize=22, markeredgecolor='k',
                     markeredgewidth=2, label='$MMD^2_u$ ($p$-value = %0.3f)'%p_value)
        plt.xlim([-0.01, 0.03])
        plt.title('$ %0.2f \leq z < %0.2f$'%(zmin, zmax), size=14)

        if i%2 == 0: ax.set_ylabel('$PDF$', size=22)
        if i > 1: ax.set_xlabel('$MMD^2_u$', size=22)
        plt.legend(numpoints=1, prop={'size':14})

    plt.savefig(plot_path + 'SDSS-vs-SV.pdf', bbox_inches='tight')

