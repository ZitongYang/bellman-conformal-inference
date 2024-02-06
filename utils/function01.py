from abc import abstractmethod
import matplotlib.pyplot as plt
import numpy as np


#############    Function0*1*   #############
class Function01:
    """
    The class of functions from [0, 1] to R
    """

    @abstractmethod
    def eval(self, input):
        pass

    def min(self, bins):
        """
        Find a minimizer of f. Note that because the domain is compact,
        the unique minimizer always exists.
        :param f: [0, 1] -> R
        :param bins: # of discretization
        :return: the exact minimizer
        """

        xs = np.linspace(1e-2, 1, bins)
        argmin = xs[np.argmin(self.eval(xs))]
        return argmin, self.eval(argmin)

    def plot(self, bins):
        """
        :param f: a function f: [0, 1] -> R
        :param bins: # of discretization
        :return: None is returned
        """
        xs = np.linspace(1e-2, 1, bins)
        plt.plot(xs, self.eval(xs))
        plt.show()

    def mean(self, bins):
        """
        :param f: a function f: [0, 1] -> R
        :param bins: # of discretization
        :return: the mean of f(bins)
        """
        xs = np.linspace(1e-2, 1, bins)
        return np.mean(self.eval(xs))


class MulFunction01(Function01):
    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2

    def eval(self, input):
        return self.f1.eval(input) * self.f2.eval(input)


class SumFunction01(Function01):
    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2

    def eval(self, input):
        return self.f1.eval(input) + self.f2.eval(input)


class DiffFunction01(Function01):
    def __init__(self, f1, f2):
        self.f1 = f1
        self.f2 = f2

    def eval(self, input):
        return self.f1.eval(input) - self.f2.eval(input)


class SymFunction01(Function01):
    def __init__(self, formula, variable, param):
        self.formula = formula
        self.variable = variable
        self.param = param

    def eval(self, input):
        exec('{} = input'.format(self.variable))
        for k, v in self.param.items():
            exec('{} = v'.format(k))
        return eval(self.formula)


class ConstantFunction01(Function01):
    def __init__(self, value):
        self.value = value

    def eval(self, input):
        try:
            return self.value * np.ones(len(input))
        except TypeError:
            return self.value


#############    Function0*T*   #############
class Function0T:
    def __init__(self, T, ys):
        self.T = T
        assert len(ys) == T + 1
        self.ys = ys

    def eval(self, inputs):
        return self.ys[inputs]

    def plot(self):
        """
        :param f: a function f: [T] -> R
        """
        plt.plot(np.array(list(range(self.T + 1))), self.ys, 'd')
        plt.show()
