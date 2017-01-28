from __future__ import division, print_function
import numpy as np

def damping_ratio(alpha, beta, w):
    return alpha/2./w + beta/2.*w

if __name__ == '__main__':

    # Resonance frequencies
    f0 = np.array([410.586, 416.983, 497.531, 826.056]) # 1369.083, 2076.187
    w0 = 2*np.pi*f0

    # Damping ratios ( 2% for all )
    zeta = np.ones(len(w0)) * 0.02

    # 2*zeta*w0 = alpha + beta*w0^2
    # zeta = alpha/2/w0 + beta/2*w0
    # zeta = A r

    A = np.empty((len(w0), 2))
    for i, wi in enumerate(w0):
        A[i, 0] = 1./2./wi
        A[i, 1] = wi/2.

    r = np.dot(np.linalg.pinv(A), zeta)

    print('zeta: ', zeta)
    print('r: ', r)
    print('damping_ratio(r): ', damping_ratio(r[0], r[1], w0))
