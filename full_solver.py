#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
#from scipy import sparse, io
#from scipy.sparse import linalg
from create_dir import create_dir
from dump2petsc import dump2petsc, read_petsc
from petsc4py import PETSc

if __name__ == '__main__':

    matrix_dir = 'matrices'
    output_dir = 'output/full'
    f_0 = 1.
    f_1 = 100.
    n_f = 1


    print('Reading matrices from disk...')
    M = read_petsc(matrix_dir+'/mass.dat', 'matrix')
    K = read_petsc(matrix_dir+'/stiffness.dat', 'matrix')
    f = read_petsc(matrix_dir+'/force.dat', 'vector')

    print('Splitting real and imaginary components...')
    # Proportional Damping matrix
    C = 0.01*M

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

    w = np.linspace(f_0*2*np.pi, f_1*2*np.pi, n_f)
    u = []
    create_dir(output_dir)

    # create solver context
    ksp = PETSc.KSP().create()
    ksp.setType('preonly')
    pc = ksp.getPC()
    pc.setType('lu')

    for wi in w:
        print('Solving for f = {:2f} Hz'.format(wi/2/np.pi))
        D = wi**2 * M1 + wi * C1 + K1
        u.append(M1.createVecRight())

        ksp.setOperators(D)
        ksp.solve(f1, u[-1])
