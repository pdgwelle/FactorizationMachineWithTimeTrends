import unittest

import numpy as np
from fmj import FlashMobJunior

class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.junior = FlashMobJunior()

    def test_default_k(self):
        junior = FlashMobJunior()
        self.assertEqual(junior.k, 4)

    def test_low_rank_interactions(self):
        n_features=5
        n_obs=10

        f = self.junior._compute_low_rank_interactions_slow

        X = np.random.binomial(n=1, p=0.0, size=(n_obs,n_features))
        Vs = np.random.rand(n_features, self.junior.k) - 0.5
        self.assertEqual(np.sum(f(X,Vs)), 0, 'all zero xs should equal 0')

        X = np.random.binomial(n=1, p=0.5, size=(n_obs,n_features))
        Vs = np.random.binomial(n=1, p=0.0, size=(n_features, self.junior.k))
        self.assertEqual(np.sum(f(X,Vs)), 0, 'all zero vs should equal 0')

        X = np.random.binomial(n=1, p=1.0, size=(n_obs,n_features))
        Vs = np.random.binomial(n=1, p=1.0, size=(n_features, self.junior.k))
        nth_triangular_number = ((n_features-1)**2 + (n_features-1)) / 2
        self.assertEqual(np.sum(f(X,Vs)), nth_triangular_number*self.junior.k*n_obs, 
            'when everything is 1 we should be able to sum easy')

        X = np.random.binomial(n=1, p=0.5, size=(n_obs,n_features))
        Vs = np.random.binomial(n=1, p=0.5, size=(n_features, self.junior.k))
        X[2,:] = np.append([1,0,1],[0]*(n_features-3))
        Vs[0,:] = [4,2,1,1]
        Vs[2,:] = [2,4,5,5]
        self.assertEqual(f(X,Vs)[2], np.dot(Vs[0,:], Vs[2,:]), 
            'dot products should work')


if __name__ == '__main__':
    unittest.main()
