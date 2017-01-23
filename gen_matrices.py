#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

from physics_classes import *
from create_dir import create_dir
from dump2petsc import dump2petsc, read_petsc
from myparams import *

import numpy as np
import sys
import os
from petsc4py import PETSc


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

    #dump2petsc(struct.M.mat(), output_dir+'/mass.dat')
    #dump2petsc(struct.K.mat(), output_dir+'/stiffness.dat')
    #dump2petsc(struct.f.vec(), output_dir+'/force.dat')

    print('Splitting real and imaginary components...')
    # Proportional Damping matrix
    M = struct.M.mat()
    K = struct.K.mat()
    C = K.duplicate(copy=True)
    C.scale(0.01)
    f = struct.f.vec()

    dump2petsc(M, output_dir+'/mass.dat')
    dump2petsc(K, output_dir+'/stiffness.dat')
    dump2petsc(C, output_dir+'/damping.dat')
    dump2petsc(f, output_dir+'/force.dat')

    '''
    # size of block
    rows = M.size[0]

    # Block mass matrix
    mi, mj, mv = M.getValuesCSR()
    mi = np.concatenate((mi[:-1], mi[:-1]+mi[-1], np.array([len(mv)*2], dtype='int32')))
    mj = np.concatenate((mj, mj+rows))
    mv = np.concatenate((-mv, -mv))
    M1 = PETSc.Mat().createAIJWithArrays(size=(2*rows, 2*rows), csr=(mi, mj, mv))

    # Block stiffness matrix
    ki, kj, kv = K.getValuesCSR()
    ki = np.concatenate((ki[:-1], ki[:-1]+ki[-1], np.array([len(kv)*2], dtype='int32')))
    kj = np.concatenate((kj, kj+rows))
    kv = np.concatenate((kv, kv))
    K1 = PETSc.Mat().createAIJWithArrays(size=(2*rows, 2*rows), csr=(ki, kj, kv))

    # Block damping matrix
    ci, cj, cv = C.getValuesCSR()
    ci = np.concatenate((ci[:-1], ci[:-1]+ci[-1], np.array([len(cv)*2], dtype='int32')))
    cj = np.concatenate((cj+rows, cj))
    cv = np.concatenate((-cv, cv))
    C1 = PETSc.Mat().createAIJWithArrays(size=(2*rows, 2*rows), csr=(ci, cj, cv))

    # Block force vector
    fv = f.getArray()
    f1 = PETSc.Vec().createWithArray(np.concatenate((fv, np.zeros(fv.shape))))


    dump2petsc(M1, output_dir+'/mass.dat')
    dump2petsc(K1, output_dir+'/stiffness.dat')
    dump2petsc(C1, output_dir+'/damping.dat')
    dump2petsc(f1, output_dir+'/force.dat')

    print("Done")
    '''
