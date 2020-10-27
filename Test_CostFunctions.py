from CostFunctions import *
import numpy as np


def test_mse():
    act_val = np.array([[0., 1., 2.]])
    est_val = np.array([[1., 0., 2.]])
    y_mse = mse(act_val, est_val)
    y_mse_grad = mse_grad(act_val, est_val)
    np.testing.assert_almost_equal(y_mse, 2./3.)
    np.testing.assert_almost_equal(y_mse_grad, np.array([[1., -1., 0.]])*2)


def test_bce_single_value():
    act_val_0 = np.array([[0.]])
    act_val_1 = np.array([[1.]])
    est_val_0 = np.array([[0.0001]])
    est_val_1 = np.array([[0.9999]])
    y_bce_act0_est0 = binary_cross_entropy(act_val_0, est_val_0)
    y_bce_act0_est1 = binary_cross_entropy(act_val_0, est_val_1)
    y_bce_act1_est0 = binary_cross_entropy(act_val_1, est_val_0)
    y_bce_act1_est1 = binary_cross_entropy(act_val_1, est_val_1)

    np.testing.assert_almost_equal(y_bce_act0_est0, -np.log(est_val_1[0][0]))
    np.testing.assert_almost_equal(y_bce_act0_est1, -np.log(est_val_0[0][0]))
    np.testing.assert_almost_equal(y_bce_act1_est0, -np.log(est_val_0[0][0]))
    np.testing.assert_almost_equal(y_bce_act1_est1, -np.log(est_val_1[0][0]))

    y_bce_grad_act0_est0 = binary_cross_entropy_grad(act_val_0, est_val_0)
    y_bce_grad_act0_est1 = binary_cross_entropy_grad(act_val_0, est_val_1)
    y_bce_grad_act1_est0 = binary_cross_entropy_grad(act_val_1, est_val_0)
    y_bce_grad_act1_est1 = binary_cross_entropy_grad(act_val_1, est_val_1)

    np.testing.assert_almost_equal(y_bce_grad_act0_est0, 1.0001)
    np.testing.assert_almost_equal(y_bce_grad_act0_est1, 10000)
    np.testing.assert_almost_equal(y_bce_grad_act1_est0, -10000)
    np.testing.assert_almost_equal(y_bce_grad_act1_est1, -1.0001)


def test_bce_matrix():
    act_val = np.array([[1., 1.], [1., 1.]])
    est_val = np.array([[0.9999, 0.9999], [0.0001, 0.9999]])
    y_bce = binary_cross_entropy(act_val, est_val)
    np.testing.assert_almost_equal(y_bce, (-np.log(0.9999)*3 - np.log(0.0001))/4)

    y_bce_grad_act0_est0 = binary_cross_entropy_grad(act_val, est_val)
    np.testing.assert_almost_equal(y_bce_grad_act0_est0, [[(-10000 - 1.0001)/2, -1.0001]])


