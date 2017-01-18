#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

from physics_classes import *
from create_dir import create_dir
from dump2petsc import dump2petsc, read_petsc
from myparams import *

import numpy as np
from scipy import sparse, io
import sys
import os


def write_vectors_pvd(physics, vectors, filename, **kwargs):

    if 'dirname' in kwargs:
        dirname = kwargs['dirname']
    else:
        dirname = '.'

    V = physics.get_function_space()
    func = Function(V)

    create_dir(dirname)
    fileh = File(dirname+'/'+filename+'.pvd')

    if 'keys' in kwargs:
        for vec, key in zip(vectors, kwargs['keys']):
            func.vector()[:] = vec
            fileh << (func, key)
    else:
        for vec in vectors:
            func.vector()[:] = vec
            fileh << func



if __name__ == '__main__':


    solid_props = {'E': E_s, 'nu': nu_s, 'rho': rho_s}
    solid_dirichlet_bc = {'values': solid_dirichlet_values, 'boundaries': solid_dirichlet_boundary_id}

    if solid_boundary_load_flag:
        solid_load = {'values': solid_load_values, 'boundaries': solid_load_boundary_id}
    else:
        solid_load = {'values': [], 'boundaries': []}

    if not solid_body_force_flag:
        solid_body_force = None

    struct = LinearSolid(grid_solid, degree_solid, solid_dirichlet_bc, solid_props, natural_bc=solid_load, body_force=solid_body_force)

    struct.assemble()

    print("Writing matrices to disk...")

    output_dir = 'matrices'
    create_dir(output_dir)

    #M = sparse.csr_matrix(struct.M.mat().getValuesCSR()[::-1])
    #K = sparse.csr_matrix(struct.K.mat().getValuesCSR()[::-1])
    #io.mmwrite(output_dir+"/mass.mtx", M)
    #io.mmwrite(output_dir+"/stiffness.mtx", K)
    #np.save(output_dir+"/force", struct.f.vec().getArray())

    dump2petsc(struct.M.mat(), output_dir+'/mass.dat')
    dump2petsc(struct.K.mat(), output_dir+'/stiffness.dat')
    dump2petsc(struct.f.vec(), output_dir+'/force.dat')


    print("Done")
