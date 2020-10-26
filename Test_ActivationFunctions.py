from ActivationFunctions import *
import numpy as np

x1 = np.ones((3, 1))
x2 = np.array([-1, 2, -3])


def test_sigmoid_alpha1():
    np.testing.assert_almost_equal(sigmoid(x1), [[0.73105858], [0.73105858], [0.73105858]])
    np.testing.assert_almost_equal(sigmoid(x2), [0.26894142, 0.880797078, 0.047425873])
    np.testing.assert_almost_equal(sigmoid_grad(x1), [[0.196611933], [0.196611933], [0.196611933]])
    np.testing.assert_almost_equal(sigmoid_grad(x2), [0.19661193, 0.104993585, 0.045176697])


def test_sigmoid_alpha3():
    alpha = 3
    np.testing.assert_almost_equal(sigmoid(x1, alpha), [[0.9525741268], [0.9525741268], [0.9525741268]])
    np.testing.assert_almost_equal(sigmoid(x2, alpha), [0.047425873178, 0.9975273768, 1.233945759E-4])
    np.testing.assert_almost_equal(sigmoid_grad(x1, alpha), [[0.04517666], [0.04517666], [0.04517666]])
    np.testing.assert_almost_equal(sigmoid_grad(x2, alpha), [0.04517666, 0.002466509, 0.000123379])


def test_relu():
    np.testing.assert_almost_equal(relu(x1), [[1], [1], [1]])
    np.testing.assert_almost_equal(relu(x2), [0, 2, 0])
    np.testing.assert_almost_equal(relu_grad(x1), [[1], [1], [1]])
    np.testing.assert_almost_equal(relu_grad(x2), [0, 1, 0])


def test_tanh():
    np.testing.assert_almost_equal(tanh(x1), [[0.76159416], [0.76159416], [0.76159416]])
    np.testing.assert_almost_equal(tanh(x2), [-0.76159416, 0.96402758, -0.99505475])
    np.testing.assert_almost_equal(tanh_grad(x1), [[0.4199743416], [0.4199743416], [0.4199743416]])
    np.testing.assert_almost_equal(tanh_grad(x2), [0.4199743416, 0.07065082, 0.009866037])
