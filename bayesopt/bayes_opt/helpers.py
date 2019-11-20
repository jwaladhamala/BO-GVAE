from __future__ import division
from __future__ import print_function

from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# import nlopt


def acq_max(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling 1e5 points at random, and then
    running L-BFGS-B from 250 random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.


    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    opt_type = 'l_bfgs_b'
    num_ini = 10
    num_ini_exp = 500;

    # Warm up with random points  
    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(num_ini_exp, bounds.shape[0]))
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(num_ini, bounds.shape[0]))

    # use either l_bfgs_b or BOBYQA for optimization
    if opt_type == 'l_bfgs_b':
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B")
            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        return np.clip(x_max, bounds[:, 0], bounds[:, 1])
    else: # use BOBYQA to optimize the parameters

        #        opt = nlopt.opt(nlopt.LN_BOBYQA,bounds.shape[0])
        #        opt.set_min_objective(lambda x,grad: float(-ac(x.reshape(1, -1), gp=gp, y_max=y_max)))
        #        opt.set_lower_bounds(bounds[:, 0])
        #        opt.set_upper_bounds(bounds[:, 1])
        #        opt.set_ftol_rel(1)
        #        opt.set_xtol_rel(10^-2)
        #        opt.set_maxeval(5000)
        #
        #        for x_try in x_seeds:
        #            # Find the minimum of minus the acquisition function
        #            try:
        #                res_x = opt.optimize(x_try)
        #                res_f = opt.last_optimum_value()
        #                if max_acq is None or -res_f >= max_acq:
        #                    x_max = res_x
        #                    max_acq = -res_f
        #
        #            except nlopt.RoundoffLimited:
        #                continue
        #                print("Oops! RoundoffLimited exception occured. Next Entry!")
        # Store it if better than previous minimum(maximum).

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        return np.clip(x_max, bounds[:, 0], bounds[:, 1])


class UtilityFunction(object):
    """
    An object to compute the acquisition functions.

    Args:
        surrogate: surrogate model used with BO
        kappa: modulate the uncertainty in UCB
        xi: a small positive term to be added to fmax in EI
        z_m: mean of p(z) or q(z) to constrain EI
        z_v: invariance variance of p(z) or q(z) to constrain EI
        z_a: weights in GMM to represent q(z)
    """

    def __init__(self, kind, kappa, z_a, z_m, z_v, xi, surrogate='gp'):
        self.surrogate = surrogate
        self.kappa = kappa
        self.xi = xi
        self.z_m = z_m
        self.z_v = z_v
        self.z_a = z_a

        if kind not in ['ei_prior', 'ei_post_agg', 'ei_post_k', 'var', 'ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of var, ucb, poi, "\
                  " ei, ei_prior, ei_post_k or ei_post_agg .".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):

        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'ei_prior':
            return self._ei_prior(x, gp, y_max, self.xi)
        if self.kind == 'ei_post_k':
            return self._ei_post_k(x, gp, y_max, self.z_a, self.z_m, self.z_v, self.xi)
        if self.kind == 'ei_post_agg':
            return self._ei_post_agg(x, gp, y_max, self.z_m, self.z_v, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)
        if self.kind == 'var':
            return self._var(x, gp)

    @staticmethod
    def _ucb(x, gp, kappa):
        # kappa= np.sqrt(0.2*log(D^(2+length(xs)/2)*pi^2/(3*0.1)));
        mean, std = gp.predict(x, return_std=True)
        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        std = std + 10**(-15)
        z = (mean - y_max - xi) / (std)
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _ei_post_agg(x, gp, y_max, z_m, z_v, xi):
        # ei is constrained with q(z) approximated
        # using a single Gaussian distribution
        # TODO: replace with matrix multiplication
        if x.shape[0] > 1:
            epi = np.zeros((x.shape[0]))
            for i in range(x.shape[0]):
                epi_temp = (x[i, :] - z_m) @ z_v @ (x[i, :] - z_m).reshape(-1, 1)
                epi[i] = epi_temp[0]
        else:
            epi = (x - z_m) @ z_v @ (x - z_m).reshape(-1, 1)
            epi = epi[0]

        tau = y_max * (1 - xi * epi)
        mean, std = gp.predict(x, return_std=True)
        std = std + 10 ** (-15)
        z = (mean - tau) / std
        return (mean - tau) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _ei_post_k(x, gp, y_max, z_a, z_m, z_v, xi):
        # ei is constrained with q(z) approximated as GMM with K components
        if x.shape[0] > 1:
            epi = np.zeros((x.shape[0]))
            for i in range(x.shape[0]):
                epi_t = 0
                for j in range(z_m.shape[1]):
                    epi_t = epi_t + z_a[:, j] * (x[i, :] - z_m[:, j]) @ z_v[:, :, j] \
                            @ (x[i, :] - z_m[:, j]).reshape(-1, 1)
                epi[i] = epi_t[0]
        else:
            epi = 0
            for j in range(z_m.shape[1]):
                epi = epi + z_a[:, j] * (x - z_m[:, j]) @ z_v[:, :, j] @ (x - z_m[:, j]).reshape(-1, 1)
            epi = epi[0]

        tau = y_max * (1 - xi * epi)
        mean, std = gp.predict(x, return_std=True)
        z = (mean - tau) / std
        return (mean - tau) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _ei_prior(x, gp, y_max, xi):
        tau = y_max * (1 - xi * np.sum(x ** 2))
        mean, std = gp.predict(x, return_std=True)
        z = (mean - tau) / std
        return (mean - tau) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, std = gp.predict(x, return_std=True)
        z = (mean - y_max - xi) / std
        return norm.cdf(z)

    @staticmethod
    def _ucb_deepgp(x, gp, kappa):
        mean, var = gp.predict(x)
        std = np.sqrt(var)
        return mean + kappa * std


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog(object):

    def __init__(self, params):

        self.ymax = None
        self.xmax = None
        self.params = params
        self.ite = 1

        self.start_time = datetime.now()
        self.last_round = datetime.now()

        # sizes of parameters name and all
        self.sizes = [max(len(ps), 7) for ps in params]

        # Sorted indexes to access parameters
        self.sorti = sorted(range(len(self.params)),
                            key=self.params.__getitem__)

    def reset_timer(self):
        self.start_time = datetime.now()
        self.last_round = datetime.now()

    def print_header(self, initialization=True):

        if initialization:
            print("{}Initialization{}".format(BColours.RED,
                                              BColours.ENDC))
        else:
            print("{}Bayesian Optimization{}".format(BColours.RED,
                                                     BColours.ENDC))

        print(BColours.BLUE + "-" * (29 + sum([s + 5 for s in self.sizes])) +
              BColours.ENDC)

        print("{0:>{1}}".format("Step", 5), end=" | ")
        print("{0:>{1}}".format("Time", 6), end=" | ")
        print("{0:>{1}}".format("Value", 10), end=" | ")

        for index in self.sorti:
            print("{0:>{1}}".format(self.params[index],
                                    self.sizes[index] + 2),
                  end=" | ")
        print('')

    def print_step(self, x, y, warning=False):

        print("{:>5d}".format(self.ite), end=" | ")

        m, s = divmod((datetime.now() - self.last_round).total_seconds(), 60)
        print("{:>02d}m{:>02d}s".format(int(m), int(s)), end=" | ")

        if self.ymax is None or self.ymax < y:
            self.ymax = y
            self.xmax = x
            print("{0}{2: >10.5f}{1}".format(BColours.MAGENTA,
                                             BColours.ENDC,
                                             y),
                  end=" | ")

            for index in self.sorti:
                print("{0}{2: >{3}.{4}f}{1}".format(
                    BColours.GREEN, BColours.ENDC,
                    x[index],
                    self.sizes[index] + 2,
                    min(self.sizes[index] - 3, 6 - 2)
                ),
                    end=" | ")
        else:
            print("{: >10.5f}".format(y), end=" | ")
            for index in self.sorti:
                print("{0: >{1}.{2}f}".format(x[index],
                                              self.sizes[index] + 2,
                                              min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")

        if warning:
            print("{}Warning: Test point chose at "
                  "random due to repeated sample.{}".format(BColours.RED,
                                                            BColours.ENDC))

        print()

        self.last_round = datetime.now()
        self.ite += 1

    def print_summary(self):
        pass
