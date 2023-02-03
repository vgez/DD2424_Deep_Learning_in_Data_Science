import unittest

import numpy as np

from assign2.assignment2 import *


class TestModel(unittest.TestCase):

    def setUp(self):
        dim = 20
        dp = 2
        self.net = Network([['data_batch_1']], 0.5, [dim, 50, 10], 0, 0.001)
        self.eps = 10**-6
        P = self.net.evaluateClassifier(self.net.train.X[:dim, :dp], self.net.W, self.net.b)
        self.gradW_a, self.gradB_a = self.net.computeGradients(P[:dim, :dp], self.net.train.Y[:dim, :dp], dp)
        self.gradW_n, self.gradB_n = self.net.computeGradsNumSlow(
            self.net.train.X[:dim, :dp], self.net.train.Y[:dim, :dp], 10**-5)
        self.checkW, self.checkB = self.net.gradientCheck(
            self.gradW_a, self.gradW_n, self.gradB_a, self.gradB_n, self.eps)

    def test_gradientMean(self):
        for i in range(len(self.gradW_a)):
            self.assertAlmostEqual(self.gradW_n[i].mean(), self.gradW_a[i].mean(), places=7)
            self.assertAlmostEqual(self.gradB_n[i].mean(), self.gradB_a[i].mean(), places=7)

    def test_relError(self):
        for i in range(len(self.checkW)):
            self.assertLessEqual(np.max(self.checkW[i]), self.eps)
            self.assertLessEqual(self.checkW[i].mean(), self.eps)
            self.assertLessEqual(np.max(self.checkB[i]), self.eps)
            self.assertLessEqual(self.checkB[i].mean(), self.eps)
