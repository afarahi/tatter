from __future__ import absolute_import, division, print_function
import numpy as np

def KL_divergence_estimator(X, Y, k=1):
    """ Estimate symmetric version of KL divergence. 
    The symmetric version is 0.5 * [ D(P|Q) + D(Q|P) ].
     
    Parameters
    ----------
        X, Y: numpy array
            2-dimensional array where each row is a sample.
        k: int optional
            k-NN to be used. The default is k=1.
    
    return
    ------
    numpy array : estimated D(P|Q)
    """
    if not (isinstance(k, int) or k is None):
        raise ValueError('k has incorrect type.')
    if k is not None and k <= 0:
        raise ValueError('k cannot be <= 0')

    kl1 =  KL_divergence_estimator_sub(X, Y, k=k)
    kl2 =  KL_divergence_estimator_sub(Y, X, k=k)
    
    return (kl1 + kl2) / 2.0


def KL_divergence_estimator_sub(X, Y, k=1):
    """ KL-Divergence estimator universal k-NN estimator.
    
    Parameters
    ----------
        X, Y: numpy array
            2-dimensional array where each row is a sample.
        k: int, optional
            k-NN to be used. The default is k=1.

    return
    ------
    numpy array : estimated D(P|Q)
    """
    n, m = len(X), len(Y)
    D = np.log(m / (n - 1))
    d = float(X.shape[1])

    for xi in X:
        nu = knn_distance(xi, Y, k-1)
        rho = knn_distance(xi, X, k)
        D += (d/n)*np.log(nu/rho)

    return D


def knn_distance(point, sample, k):
    """ Euclidean distance from `point` to it's `k`-Nearest
    Neighbour in `sample`.
    
    Parameters
    ----------
        point: float
            a data point from a sample.
        sample: numpy array
            2-dimensional array where each row is a sample.
        k: int
            k-NN to be used.

    return
    ------
    numpy-array : `k`-Nearest Neighbour in `sample` 
    """
    norms = np.linalg.norm(sample-point, axis=1)
    return np.sort(norms)[k]

