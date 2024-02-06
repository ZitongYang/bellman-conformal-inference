from utils.function01 import Function01, Function0T
from utils import argmin01, positive_part
from utils.function01 import ConstantFunction01, \
                             MulFunction01
import matplotlib.pyplot as plt
import numpy as np


class DynamicConformal:
    """
    The abstract class that computes and visualize the dynamic programming
    algorithm for conformal inference.
    """
    def __init__(self, horizon, alpha0, disturbances, costs, lbd, rhoTp, Tp):
        """
        :param horizon: <int>
        :param alpha0: <double> desired miscoverage level
        :param disturbances: <Function01 List> of length T, the CDFs of beta
        :param costs: <Function01 List> of length T
        :param lbd: <double> multiplicative weight to terminal coverage cost
        :param rhoTp: <int> the number of mistakes in prev. Tp steps
        :param Tp: <int> size of the look-back window
        """
        self.T = horizon
        self.disturbances = disturbances
        self.costs = costs
        self.alpha0 = alpha0
        self.lbd = lbd
        self.rhoTp = rhoTp
        self.Tp = Tp

        self.optimal_policy = None
        self.cost_to_go = None

    def dp(self, bins):
        """
        Using dynamic programming to solve the optimal policy
        :param bins: <int> Discretization accuracy for interval [0, 1]
        """
        cost_to_go = list(range(self.T+1))
        optimal_policy = list(range(self.T))
        states = np.array(list(range(self.T+1)))

        # the cost to go function for the T+1-th round
        cost_to_go[self.T] = Function0T(
            self.T,
            self.lbd*positive_part((states+self.rhoTp)/(self.T+self.Tp) - self.alpha0)
        )
        # the t here means t-th round. Because of the 0-indexing convention
        # of python, we use list[t-1] when indexing an object at t-th round

        # for t in tqdm(range(self.T, 0, -1)): # show progress bar for debugging
        for t in range(self.T, 0, -1):  # t = T, T-1, T-2, ... , 1
            L = self.costs[t-1]
            Jtp1 = cost_to_go[t]
            D = Function0T(Jtp1.T-1, Jtp1.ys[1:] - Jtp1.ys[:Jtp1.T])
            F = self.disturbances[t-1]
            optimal_policy_temp, cost_to_go_temp = argmin01(L, D, F, bins)

            # storing the DP data
            optimal_policy[t-1] = optimal_policy_temp
            cost_to_go_temp = Function0T(cost_to_go_temp.T,
                                         cost_to_go_temp.ys + Jtp1.ys[:Jtp1.T])
            cost_to_go[t-1] = cost_to_go_temp

        self.optimal_policy = optimal_policy
        self.cost_to_go = cost_to_go

    def visualize_optimal_policy(self):
        fig, axs = plt.subplots(2)
        fig.suptitle('Optimal policies')
        plot_index = list(range(0, self.T, int(self.T/5)))
        for t in plot_index:
            axs[0].plot(range(self.optimal_policy[t].T+1), self.optimal_policy[t].ys,
                        'd--', label='t={}'.format(t + 1))
            axs[1].plot(range(self.cost_to_go[t].T+1), self.cost_to_go[t].ys,
                        'd--', label='t={}'.format(t + 1))
        axs[0].set(ylabel=r'$\mu_t(\rho)$')
        axs[1].set(ylabel=r'$J_t(\rho$)', xlabel=r'$\rho$')
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
        plt.show()

    def visualize_dynamic_system(self):
        fig, axs = plt.subplots(2)
        fig.suptitle('Cost functions')
        plot_index = list(range(0, self.T, int(self.T/5)))
        zointerval = np.linspace(0, 1, 200)
        for t in plot_index:
            axs[0].plot(zointerval, self.costs[t].eval(zointerval),
                        label='t={}'.format(t + 1))
            axs[1].plot(zointerval, self.disturbances[t].eval(zointerval),
                        label='t={}'.format(t + 1))
        axs[0].set(ylabel=r'$L_t(\alpha)$')
        axs[1].set(ylabel=r'$F_t(\alpha$)', xlabel=r'$\alpha$')
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
        plt.show()

    def visualize_lambda(self):
        binsize = 100
        oldlbd = self.lbd
        lbds = oldlbd * np.linspace(0, 5, binsize)
        alphas = np.zeros(binsize)
        for i in range(len(lbds)):
            self.lbd = lbds[i]
            self.dp(bins=200)
            alphas[i] = self.optimal_policy[0].eval(0)

        # plotting
        plt.plot(lbds, alphas, color='cornflowerblue', linewidth=2)
        plt.plot(lbds, alphas*0, '--', color='salmon', linewidth=2)
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\alpha$')
        plt.show()

        self.lbd = oldlbd
