from __future__ import division
from __future__ import print_function

from math import sqrt, pi, exp
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

TOLERANCE = 1e-5    # relative error at which curve-fit is accepted
MAX_ITER = 10       # maximum number of iterations in piecewise fitting
MIN_SPLIT_LEN = 2  # minimum size of the pieces in piecewise fitting
THRESHOLD = 1e-12   # threshold value for weights. Anything less than this is assigned zero value


class Fitted:

    def __init__(self):
        self.tol = TOLERANCE
        self.weight_eps = 1e-2

    def set_tol(self, tt):
        self.tol = tt

    def set_coeffs(self, w_c, f1_c, f2_c):
        self.w = w_c
        self.c1 = f1_c
        self.c2 = f2_c

    def set_errs(self, f1_emax, f1_emax_i, f2_emax, f2_emax_i):
        self.e1 = f1_emax
        self.e1_i = f1_emax_i
        self.e2 = f2_emax
        self.e2_i = f2_emax_i

    def is_tol(self):
        if max(self.e1, self.e2) < self.tol:
            return True
        else:
            return False

    def get_err_max(self):
        if self.e1 > self.e2:
            return (self.e1, self.e1_i)
        else:
            return (self.e2, self.e2_i)

def fit_aux(w_range, func1, func2, func_fit):
    #f1 = alpha_v/np.sqrt(w_range)
    #f2 = alpha_h*w_range*np.sqrt(w_range)
    f1 = func1(w_range)
    f2 = func2(w_range)

    popt1, pcov1 = curve_fit(func_fit, w_range, f1)
    popt2, pcov2 = curve_fit(func_fit, w_range, f2)

    f1_fit = func_fit(w_range, popt1[0], popt1[1], popt1[2])
    f2_fit = func_fit(w_range, popt2[0], popt2[1], popt2[2])

    f1_err = np.abs(1-f1_fit/f1)
    f2_err = np.abs(1-f2_fit/f2)

    f1_err_max = np.max(f1_err)
    f1_err_max_ind = np.argmax(f1_err)

    f2_err_max = np.max(f2_err)
    f2_err_max_ind = np.argmax(f2_err)

    fitted = Fitted()
    fitted.set_coeffs(w_range, popt1, popt2)
    fitted.set_errs(f1_err_max, f1_err_max_ind, f2_err_max, f2_err_max_ind)

    return fitted


def greedy_fit(omega, f1, f2, fit_func):
    split_index = [0, len(omega)-1]
    fit_list = []

    n_iter = 0
    ii = 1
    fit = None
    while ii < len(split_index):
        w = omega[split_index[ii-1]:split_index[ii]]

        if len(w) <= MIN_SPLIT_LEN:
            print('Split too small (length < {}). Moving on'.format(MIN_SPLIT_LEN))
            if fit:
                if fit not in fit_list:
                    fit_list.append(fit)
                split_index.pop(ii)
                ii += 1
                n_iter = 0
            else:
                break
        elif n_iter > MAX_ITER:
            print('Exceeded MAX_ITER in split. Moving on. Max. error = {}'.format(fit.get_err_max()[0]))
            if fit not in fit_list:
                fit_list.append(fit)
            split_index.pop(ii)
            ii += 1
            n_iter = 0
        else:
            fit = fit_aux(w, f1, f2, fit_func)
            n_iter += 1

            if fit.is_tol():
                print('Max. error below TOLERANCE. Moving on')
                fit_list.append(fit)
                ii += 1
                n_iter = 0
            else:
                split_index.insert(ii, split_index[ii-1] + len(w)//2)
                #split_index.append(split_index[ii-1] + len(w)//2)
                #split_index.sort()


    return fit_list

def get_weights(w_tot, centres):
    beta = 25
    #sigma = 0.1
    weights = np.zeros((len(w_tot), len(centres)))
    for i, w in enumerate(w_tot):
        d_i = np.abs(centres-w)
        m = np.min(d_i)
        if m == 0:
            j = np.argmin(d_i)
            weights[i,:] = np.zeros(d_i.shape)
            weights[i,j] = 1
        else:
            weights[i,:] = np.exp(-beta*d_i/m)
        #weights[i,:] = np.exp(-d_i**2/sigma**2)
            weights[i,:] = weights[i,:]/np.sum(weights[i,:])

    weights[weights<THRESHOLD] = 0
    return weights

if __name__ == "__main__":

    f_0 = 300
    f_1 = 600
    n_f = 150

    omega = np.linspace(f_0*2*pi, f_1*2*pi, n_f)

    mu = 1e-2

    def func1(x):
        #s = 1j*x
        #return np.real(mu/(s+mu))
        return mu/(mu**2 + x**2)

    def func2(x):
        #s = 1j*x
        #return np.imag(mu/(s+mu))
        return -mu*x/(mu**2 + x**2)

    def func_fit(x, e0, e1, e2):
        return e0 + e1*x + e2/x
        #return e0 + e1*x + e2*x**2

    fit_list = greedy_fit(omega, func1, func2, func_fit)

    #split_limits = np.array([(xx.w[0], xx.w[-1]) for xx in fit_list])
    split_means = np.array([(xx.w[0] + xx.w[-1])/2 for xx in fit_list])

    weights = get_weights(omega, split_means)



    fig_fit = plt.figure(1)
    ax_fit = fig_fit.add_subplot(111)

    fig_err = plt.figure(2)
    ax_err = fig_err.add_subplot(111)


    f1 = func1(omega)
    f2 = func2(omega)

    f1_fit = np.zeros(omega.shape)
    f2_fit = np.zeros(omega.shape)

    for f_i, fit in enumerate(fit_list):
        f1_fit += func_fit(omega, fit.c1[0], fit.c1[1], fit.c1[2])*weights[:, f_i]
        f2_fit += func_fit(omega, fit.c2[0], fit.c2[1], fit.c2[2])*weights[:, f_i]

    f1_err = np.abs(1-f1_fit/f1)
    f2_err = np.abs(1-f2_fit/f2)

    ax_fit.semilogy(omega/2/pi, np.abs(f1), color='black', linestyle='solid')
    ax_fit.semilogy(omega/2/pi, np.abs(f2), color='gray', linestyle='solid')
    ax_fit.semilogy(omega/2/pi, np.abs(f1_fit), color='black', linestyle='dashed')
    ax_fit.semilogy(omega/2/pi, np.abs(f2_fit), color='gray', linestyle='dashed')

    ax_err.semilogy(omega/2/pi, f1_err, color='black')
    ax_err.semilogy(omega/2/pi, f2_err, color='gray')

    # for fit in fit_list:
    #     f1 = alpha_v/np.sqrt(fit.w)
    #     f2 = alpha_h*fit.w*np.sqrt(fit.w)
    # 
    #     f1_fit = func_fit(fit.w, fit.c1[0], fit.c1[1], fit.c1[2])
    #     f2_fit = func_fit(fit.w, fit.c2[0], fit.c2[1], fit.c2[2])
    # 
    #     f1_err = np.abs(1-f1_fit/f1)
    #     f2_err = np.abs(1-f2_fit/f2)
    # 
    #     ax_fit.semilogy(fit.w/2/pi, np.abs(f1), color='black', linestyle='solid')
    #     ax_fit.semilogy(fit.w/2/pi, np.abs(f2), color='gray', linestyle='solid')
    #     ax_fit.semilogy(fit.w/2/pi, np.abs(f1_fit), color='black', linestyle='dashed')
    #     ax_fit.semilogy(fit.w/2/pi, np.abs(f2_fit), color='gray', linestyle='dashed')
    # 
    #     ax_err.semilogy(fit.w/2/pi, f1_err, color='black')
    #     ax_err.semilogy(fit.w/2/pi, f2_err, color='gray')

    f_pieces = [xx.w[0]/2/pi for xx in fit_list]
    #f_pieces = [omega[i]/2/pi for i in split_index]

    ax_fit_ylims = ax_fit.get_ylim()
    ax_fit.vlines(f_pieces, ax_fit_ylims[0], ax_fit_ylims[1], colors='k', linestyles='dashdot')
    ax_fit.set_xlabel(r'Frequency [Hz]')
    ax_fit.legend([r'$|\mathcal{R}\{g({\rm i} \omega)\}|$', r'$|\mathcal{I}\{g({\rm i} \omega)\}|$'], loc='best')
    ax_fit.grid(True)

    ax_err_ylims = ax_err.get_ylim()
    ax_err.vlines(f_pieces, ax_err_ylims[0], ax_err_ylims[1], colors='k', linestyles='dashdot')
    ax_err.set_xlabel(r'Frequency [Hz]')
    ax_err.set_ylabel(r'Relative error')
    ax_fit.legend([r'$|\mathcal{R}\{g({\rm i} \omega)\}|$', r'$|\mathcal{I}\{g({\rm i} \omega)\}|$'], loc='best')
    ax_err.grid(True)

    plt.show()
