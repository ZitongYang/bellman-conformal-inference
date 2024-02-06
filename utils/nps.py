from utils.function01 import DiffFunction01, MulFunction01, ConstantFunction01

import numpy as np
import matplotlib.pyplot as plt
import warnings

class NestedPredictionInterval:
    """
    Abstract class that implements nested prediction intervals
    """
    def __init__(self, upper, lower):
        self.upper = upper
        self.lower = lower

    def cover(self, alpha, yhat):
        upper_bd = self.upper.eval(alpha)
        lower_bd = self.lower.eval(alpha)
        if np.abs(upper_bd - yhat) < 1e-3 or np.abs(lower_bd - yhat) < 1e-3:
            return 'exact cover'
        elif upper_bd > yhat and yhat > lower_bd:
            return 'over cover'
        else:
            return 'under cover'

    def beta_threshold(self, yhat):
        left = 0
        right = 1
        curr_beta = (left + right) / 2

        iter = 0
        MAX_ITER = 2000
        while self.cover(curr_beta, yhat) != 'exact cover':
            iter += 1
            if self.cover(curr_beta, yhat) == 'over cover':
                left = curr_beta
            else:
                right = curr_beta
            curr_beta = (left + right) / 2
            if curr_beta < 1e-14:
                warnings.warn("The value of beta less than 1e-14."
                              "This is effective 0 and the interval "
                              "length is infinity.")
                break
            if iter > MAX_ITER:
                warnings.warn("Maxium beta search iteration exceeded."
                              "Expect approximate results.")
                break
        return curr_beta

    def length(self):
        return DiffFunction01(self.upper, self.lower)

    def norm_length(self):
        len_func = DiffFunction01(self.upper, self.lower)
        meanlength = len_func.mean(200)
        return MulFunction01(len_func, ConstantFunction01(1/meanlength))

    def plot_bounds(self, bins=100):
        alphas = np.linspace(0, 1, bins)
        upper_bounds = self.upper.eval(alphas)
        lower_bounds = self.lower.eval(alphas)

        plt.plot(alphas, upper_bounds, label='Upper Bound')
        plt.plot(alphas, lower_bounds, label='Lower Bound')

        plt.xlabel('Alpha')
        plt.ylabel('Bounds')
        plt.legend()
        plt.show()
