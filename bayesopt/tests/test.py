# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:47:52 2018

@author: jd1336
"""

import numpy as np
from bayes_opt import BayesianOptimization


def camel6(x, vae=0):
    # min is -1.0316 (0.0898,-0.7126) and (-0.0898,0.7126); [-3,3,[-2,2]]

    x1, x2 = x[0], x[1]
    f1 = (4.0 - 2.1 * x1 ** 2 + (x1 ** 4) / 3.0) * (x1 ** 2) + (x1 * x2) + (-4 + 4 * (x2 ** 2)) * (x2 ** 2)
    return -f1


def branin(x, vae=0):
    #    print(x)
    x1, x2 = x[0], x[1]
    a, b, c = 1, 5.1 / (4 * np.pi ** 2), 5 / np.pi
    r, s, t = 6, 10, 1 / (8 * np.pi)
    return -(a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)


parUnknownId = [1, 2]
bounds = [(-5, 5) for i in parUnknownId]
parUnknownId = [str(i) for i in parUnknownId]

gp_surr = BayesianOptimization(camel6,
                               dict(zip(parUnknownId, bounds)), 0, 0)
gp_surr.maximize(init_points=10, n_iter=100, acq='ei')
