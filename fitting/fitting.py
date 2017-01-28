import numpy as np
from math import pi, sqrt
from piecewise_fit_greedy import greedy_fit, get_weights
#from myparams import *

f_0 = 300
f_1 = 900
n_f = 300

mu = 825.056

def func1(x):
    #s = 1j*x
    #return np.real(mu/(s+mu))
    return mu/(mu**2 + x**2)

def func2(x):
    #s = 1j*x
    #return np.imag(mu/(s+mu))
    return -mu*x/(mu**2 + x**2)

def func_fit(x, e0, e1, e2):
    return e0/x + e1 + e2*x
    #return e0 + e1*x + e2*x**2

omega = np.linspace(2*pi*f_0, 2*pi*f_1, n_f)

fit_list = greedy_fit(omega, func1, func2, func_fit)

split_means = np.array([(xx.w[0] + xx.w[-1])/2 for xx in fit_list])

weights = get_weights(omega, split_means)

np.savetxt('fit_list.txt', np.vstack([np.hstack((ff.c1, ff.c2)) for ff in fit_list]))
np.savetxt('weights.txt', weights)
