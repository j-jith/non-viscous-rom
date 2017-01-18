from __future__ import print_function
from petsc4py import PETSc

def dump2petsc(A, filename):
    viewer = PETSc.Viewer().createBinary(filename,'w')
    viewer(A)
    viewer.destroy()

def read_petsc(filename, data_type):
    viewer = PETSc.Viewer().createBinary(filename,'r')
    if data_type == 'matrix':
        A = PETSc.Mat()
    elif data_type == 'vector':
        A = PETSc.Vec()
    else:
        print('Unknown data type. Pass matrix or vector.')
        return None

    A.create(PETSc.COMM_WORLD)
    A.load(viewer)

    viewer.destroy()

    return A
