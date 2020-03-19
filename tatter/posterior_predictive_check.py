"""
Posterior Predictive check with Maximum Mean Discrepancy (MMD).

Introduction:
-------------
TATTER (Two-sAmple TesT EstimatoR) is equipped with a posterior predictive check that allows
the users to study the goodness of a data generative model vs. an observed data. This module employ
MMD, KL, KS test to compute the distance a set of simulated dataset.


Quickstart:
-----------
To start using TATTER, simply use "from tatter import posterior_predictive_check"
to access the posterior predictive check function. The exact requirements
for the inputs are listed in the docstring of the  posterior_predictive_check()
function further below. An example of this function looks like this:

    -----------------------------------------------------------------
    |  from tatter import posterior_predictive_check                |
    |                                                               |
    |  test_value, test_null, p_value =                             |
    |          posterior_predictive_check(X, sims,                  |
    |                                     model='MMD',              |
    |                                     kernel_function='rbf',    |
    |                                     gamma=gamma,              |
    |                                     n_jobs=4,                 |
    |                                     verbose=True,             |
    |                                     random_state=0)           |
    |                                                               |
    -----------------------------------------------------------------


Author:
--------
Arya Farahi, aryaf@umich.edu
Data Science Fellow
University of Michigan -- Ann Arbor


Libraries:
----------
The two-sample test estimator used in this implementation utilizes
'numpy', 'matplotlib', 'sklearn', 'joblib', 'tqdm', and 'pathlib' libraries.


References:
-----------
[1]. A. Gelman, X. L. Meng, and H. Stern,
    "Posterior predictive assessment of model fitness via realized discrepancies."
    Statistica sinica (1996): 733-760.
"""

from __future__ import division
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import pairwise_kernels
from .KL_estimator import KL_divergence_estimator
from .TST_estimators import MMD2u_estimator, MMD_null_estimator,\
    KS_null_estimator, KL_divergence_estimator, KL_null_estimator
from joblib import Parallel, delayed

def posterior_predictive_check(X, sims, model='MMD', kernel_function='rbf', verbose=False,
                               random_state=None, n_jobs=1, **kwargs):
    """
     This function performs a posterior predictive check. The posterior predictive check is a
      Bayesian counterpart of the classical tests for goodness-of-the-fit. It often can be used in
      judging the fit of a single Bayesian model to the observed data. The posterior predictive
      check require a test statistics. In our implementation, the test sample here can be one of
      the following tests: Kolmogorov-Smirnov, Kullback-Leibler divergence, or MMD.
      We note that the Kolmogorov-Smirnov test is defined for only one-dimensional data.
      The module perform a bootstrap algorithm to estimate the null distribution, and corresponding p-value.

    :param X: numpy-array
        Observed data, of size M x D [ is the number of data points, D is the features dimension]
    :param sims: list of numpy-array
        A list of a set of simulated data, of size N x D
        [Nsim is the number of simulatins, N is the number of data points, D is the features dimension]
    :param model: string
        defines the basis model to perform two sample test ['KS', 'KL', 'MMD']
    :param kernel_function: string
        defines the kernel function, only used for the MMD.
        For the list of implemented kernel please consult with https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics
    :param verbose: bool
        controls the verbosity of the model's output.
    :param random_state: type(np.random.RandomState()) or None
        defines the initial random state.
    :param n_jobs: int [not implimented yet]
        number of jobs to run in parallel.
    :param kwargs:
        extra parameters, these are passed to `pairwise_kernels()` as kernel parameters or `KL_divergence_estimator()`
         as the number of k. E.g., if `kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1)`

    :return: tuple of size 3
        float: the test value,
        numpy-array: a null distribution via bootstraps,
        float: estimated p-value
    """

    if model not in ['KS', 'KL', 'MMD']:
        raise ValueError("The Model '%s' is not implemented, try 'KS', 'KL', or 'MMD'."%model)

    # if len(X.shape) != 2 or len(sim.shape) != 3:
    #    raise ValueError("Incompatible shape for X and sim matrices. X and sim should have a shape of length 2 and 3 respectively,"
    #                     ": len(X.shape) == %i while len(sim.shape) == %i."%(len(X.shape), len(sim.shape)))

    # if X.shape[1] != sim.shape[1]:
    #    raise ValueError("Incompatible dimension for X and Y matrices. X and Y should have the same feature dimension,"
    #                     ": X.shape[1] == %i while Y.shape[1] == %i."%(X.shape[1], Y.shape[1]))

    if model == 'KS' and X.shape[1] > 1:
        raise ValueError("The KS test can handle only one dimensional data,"
                         ": X.shape[1] == %i and Y.shape[1] == %i."%(X.shape[1], Y.shape[1]))

    if not (isinstance(n_jobs, int) and n_jobs > 0):
        raise ValueError('n_jobs is incorrect type or <1. n_jobs:%s'%n_jobs)

    # define the random state
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    m = len(X)

    # p-value's resolution
    resolution = 1.0/len(sims)

    test_value = []
    null_value = []

    # compute the test statistics according to the input model
    # compute the null distribution via a bootstrap algorithm
    if model == 'MMD':

        for Y in sims:

            n = len(Y)
            XY = np.vstack([X, Y])
            K = pairwise_kernels(XY, metric=kernel_function, **kwargs)

            test_value += [MMD2u_estimator(K, m, n)]
            # nv = np.mean([MMD_null_estimator(K, m, n, rng) for i in range(20)])
            null_value += [MMD_null_estimator(K, m, n, rng)]

    elif model == 'KS':

        for Y in sims:

            n = len(Y)
            K = np.concatenate((X.T[0], Y.T[0]))

            test_value += [stats.ks_2samp(X.T[0], Y.T[0])[0]]
            null_value += [KS_null_estimator(K, m, n, rng)]

    elif model == 'KL':

        for Y in sims:
            n = len(Y)
            K = np.vstack([X, Y])

            test_value += [KL_divergence_estimator(X, Y, **kwargs)]
            null_value += [KL_null_estimator(K, m, n, rng)]

    test_value = np.array(test_value)
    null_value = np.array(null_value)

    if verbose:
        print("test value = %s"%test_value)
        print("Computing the null distribution.")

    # compute the p-value, if less then the resolution set it to the resolution
    p_value = max(resolution, resolution*(null_value > test_value).sum())

    if verbose:
        if p_value == resolution:
            print("p-value < %s \t (resolution : %s)" % (p_value, resolution))
        else:
            print("p-value ~= %s \t (resolution : %s)" % (p_value, resolution))

    return test_value, null_value, p_value


def test_statistics(X, Y, model='MMD', kernel_function='rbf', **kwargs):
    """
     This function performs a test statistics and return a test value. This implementation can perform
     the Kolmogorov-Smirnov test (for one-dimensional data only), Kullback-Leibler divergence and MMD.

    :param X: numpy-array
        Data, of size MxD [M is the number of data points, D is the features dimension]
    :param Y: numpy-array
        Data, of size NxD [N is the number of data points, D is the features dimension]
    :param model: string
        defines the basis model to perform two sample test ['KS', 'KL', 'MMD']
    :param kernel_function: string
        defines the kernel function, only used for the MMD.
        For the list of implemented kernel please consult with https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.kernel_metrics.html#sklearn.metrics.pairwise.kernel_metrics
    :param kwargs:
        extra parameters, these are passed to `pairwise_kernels()` as kernel parameters or `KL_divergence_estimator()`
        as the number of k. E.g., if `kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1)`

    :return: float
        the test value
    """

    if model not in ['KS', 'KL', 'MMD']:
        raise ValueError("The Model '%s' is not implemented, try 'KS', 'KL', or 'MMD'." % model)

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices. X and Y should have the same feature dimension,"
                         ": X.shape[1] == %i while Y.shape[1] == %i." % (X.shape[1], Y.shape[1]))

    if model == 'KS' and X.shape[1] > 1:
        raise ValueError("The KS test can handle only one dimensional data,"
                         ": X.shape[1] == %i and Y.shape[1] == %i." % (X.shape[1], Y.shape[1]))

    m = len(X)
    n = len(Y)

    # compute the test statistics according to the input model
    if model == 'MMD':
        XY = np.vstack([X, Y])
        K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
        test_value = MMD2u_estimator(K, m, n)

    elif model == 'KS':
        test_value, _ = stats.ks_2samp(X.T[0], Y.T[0])

    elif model == 'KL':
        test_value = KL_divergence_estimator(X, Y, **kwargs)

    return test_value
