import unittest

import numpy as np

from code.assignment3 import *


class TestModel(unittest.TestCase):

    def setUp(self):
        dim = 20
        dp = 5
        train_data, _, _ = getData([['data_batch_1']], 0.5, 10)
        self.net = Network(layers=[dim, 50, 50, 10], lmbda=0, eta=0.001, batchNorm=True)
        self.eps = 1e-5
        P = self.net.evaluateClassifier(train_data.X[:dim, :dp], self.net.W, self.net.b)
        self.gradW_a, self.gradB_a, self.gradGamma_a, self.gradBeta_a = self.net.computeGradients(P[:dim, :dp], train_data.Y[:dim, :dp], dp)
        self.gradW_n, self.gradB_n, self.gradGamma_n, self.gradBeta_n = self.net.computeGradsNumSlow(train_data.X[:dim, :dp], train_data.Y[:dim, :dp], 1e-5)
        self.checkW = self.net.gradientCheck(self.gradW_a, self.gradW_n, self.eps)
        self.checkB = self.net.gradientCheck(self.gradB_a, self.gradB_n, self.eps)
        self.checkGamma = self.net.gradientCheck(self.gradGamma_a, self.gradGamma_n, self.eps)
        self.checkBeta = self.net.gradientCheck(self.gradBeta_a, self.gradBeta_n, self.eps)

    def test_gradientMean(self):
        for i in range(len(self.gradW_a)):
            self.assertAlmostEqual(self.gradW_n[i].mean(), self.gradW_a[i].mean(), places=7)
            self.assertAlmostEqual(self.gradB_n[i].mean(), self.gradB_a[i].mean(), places=7)

        for i in range(len(self.gradGamma_a)):
            self.assertAlmostEqual(self.gradGamma_n[i].mean(), self.gradGamma_a[i].mean(), places=7)
            self.assertAlmostEqual(self.gradBeta_n[i].mean(), self.gradBeta_a[i].mean(), places=7)

    def test_relError(self):
        for i in range(len(self.checkW)):
            self.assertLessEqual(np.max(self.checkW[i]), self.eps)
            self.assertLessEqual(self.checkW[i].mean(), self.eps)
            self.assertLessEqual(np.max(self.checkB[i]), self.eps)
            self.assertLessEqual(self.checkB[i].mean(), self.eps)

        for i in range(len(self.checkGamma)):
            self.assertLessEqual(np.max(self.checkGamma[i]), self.eps)
            self.assertLessEqual(self.checkGamma[i].mean(), self.eps)
            self.assertLessEqual(np.max(self.checkBeta[i]), self.eps)
            self.assertLessEqual(self.checkBeta[i].mean(), self.eps)

if __name__=='__main__':
	unittest.main()

