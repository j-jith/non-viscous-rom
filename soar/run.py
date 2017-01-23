#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

from numpy import linspace, around
#from math import pi
import os

# frequency array

f_0 = 150
f_1 = 950

# length of frequency array
n_f = 801

f_arr = linspace(f_0, f_1, n_f)

# number of interpolation points
n_ip = 20

# number of arnoldi basis vectors per interpolation point
n_arn = 20

# number of processes to be run in paralle
n_p = 4

# calculating the index of interpolation points (evenly distributed)
ind1 = linspace(0, n_f-1, n_ip+1, endpoint=True)
ind1_mid = (ind1[:-1] + ind1[1:])/2
ip_ind = around(ind1_mid, decimals=0).astype(int)

# calling the SOAR program
for i in range(n_ip):

    print('Interpolation point {}/{}: Index = {}, Value = {} Hz.'.format(i+1, n_ip, ip_ind[i], f_arr[ip_ind[i]]))

    os.system("mpirun -np {} ./soar {} {}".format(n_p, ip_ind[i], n_arn))
