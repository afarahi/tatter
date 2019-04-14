
import sys
sys.path.insert(0, sys.path[0]+'/tests')

def main():

    import time
    import warnings
    from tests import \
        test_gaussian_vs_lognormal, \
        RM_clusters_consistency_check, \
        RM_clusters_witness_function, \
        mnist_digits_distance

    warnings.filterwarnings("ignore")

    print("Start ... ")

    start = time.time()

    # test_gaussian_vs_lognormal() ## [passed]
    # RM_clusters_consistency_check() ## [passed]
    # RM_clusters_witness_function() ## [passed]
    mnist_digits_distance() ## [passed]

    end = time.time()
    print("The execution time is %0.2f (s)" % (end - start))

    print("... end. ")


if __name__ == '__main__':
    main()
