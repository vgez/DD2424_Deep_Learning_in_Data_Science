import unittest

import numpy as np

from assign1.assignment1 import *


class TestModel(unittest.TestCase):

    def setUp(self):
        self.train = np.array(separate(loadBatch('test_batch'), 10))
        self.W = np.random.normal(0, 0.01, (10, 3072))
        self.b = np.random.normal(0, 0.01, (10, 1))
        self.lmda = 0
        self.P = evaluateClassifier(self.train[0], self.W, self.b)
        dp = 10
        self.gradW_n, self.gradB_n = computeGradsNumSlow(
            self.train[0][:, :dp], self.train[1][:, :dp], self.W, self.b, self.lmda, 10**-6)
        self.gradW_a, self.gradB_a = computeGradients(
            self.P[:, :dp], self.train[0][:, :dp], self.train[1][:, :dp], self.W, self.lmda, dp)
        self.checkW, self.checkB = gradientCheck(self.gradW_a, self.gradW_n, self.gradB_a, self.gradB_n, 10**-6)

    def test_gradientMean(self):
        self.assertAlmostEqual(self.gradW_n.mean(), self.gradW_a.mean(), places=7)
        self.assertAlmostEqual(self.gradB_n.mean(), self.gradB_a.mean(), places=7)

    def test_relError(self):
        self.assertLessEqual(np.max(self.checkW), 10**-6)
        self.assertLessEqual(self.checkW.mean(), 10**-6)
        self.assertLessEqual(np.max(self.checkB), 10**-6)
        self.assertLessEqual(self.checkB.mean(), 10**-6)
