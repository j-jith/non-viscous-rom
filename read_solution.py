#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import os
import sys

from petsc4py import PETSc
from dolfin import *
import numpy as np

from myparams import *
from physics_classes import *

from create_dir import create_dir

if __name__ == '__main__':

    # # viscous full solution
    # solution_dir = 'output/full/'
    # solution_prefix = 'solution_'
    # output_file = 'output/point_full.csv'
    # output_dir = 'output/plot/'

    # non-viscous full solution
    solution_dir = 'soar/output/full/solution/'
    solution_prefix = ''
    output_file = 'soar/output/point_full.csv'
    output_dir = 'soar/output/full/plot/'

    # # SOAR
    # solution_dir = 'soar/output/red/solution/'
    # solution_prefix = ''
    # output_file = 'soar/output/point_red.csv'
    # output_dir = 'soar/output/red/plot/'

    solution_n = 300 # total number of solutions
    solution_point = np.array([0.12, 0., 0.01])
    solution_axis = 2

    solid_props = {'E': E_s, 'nu': nu_s, 'rho': rho_s}
    solid_dirichlet_bc = {'values': solid_dirichlet_values, 'boundaries': solid_dirichlet_boundary_id}

    if solid_boundary_load_flag:
        solid_load = {'values': solid_load_values, 'boundaries': solid_load_boundary_id}
    else:
        solid_load = {'values': [], 'boundaries': []}

    if not solid_body_force_flag:
        solid_body_force = None

    struct = LinearSolid(grid_solid, degree_solid, solid_dirichlet_bc, solid_props, natural_bc=solid_load, body_force=solid_body_force)

    sol_real = Function(struct.get_function_space())
    sol_imag = Function(struct.get_function_space())

    # get solution vectors
    #filenames = []
    #for f in os.listdir(solution_dir):
    #    if f.endswith('.dat'):
    #        filenames.append(f)

    #filenames = sorted(filenames)

    pp = Point(solution_point[0], solution_point[1], solution_point[2])
    point_file = open(output_file, 'w')
    u_binary = PETSc.Vec()
    u_binary.create(PETSc.COMM_WORLD)

    create_dir(output_dir)

    real_file = File(output_dir+'solution_real.pvd')
    imag_file = File(output_dir+'solution_imag.pvd')

    #for i, f in enumerate(filenames):
    for i in range(solution_n):
        #print('Reading {}...'.format(f))
        #viewer = PETSc.Viewer().createBinary(solution_dir+f,'r')
        print('Reading solution {}...'.format(i))
        viewer = PETSc.Viewer().createBinary(solution_dir +
                solution_prefix + '{}.dat'.format(i),'r')
        u_binary.load(viewer)
        viewer.destroy()

        ll = u_binary.getSize()
        u_array = u_binary.getArray()

        sol_real.vector()[:] = u_array[:ll//2]
        sol_imag.vector()[:] = u_array[ll//2:]

        point_file.write('{}, {}, {}\n'.format(i, sol_real(pp)[solution_axis], sol_imag(pp)[solution_axis]))

        real_file << sol_real
        imag_file << sol_imag

    point_file.close()


'''
# assign petsc binary vector to function
u.vector()[:] = u_binary.getArray(True)
#u_petsc = as_backend_type(u.vector()).vec()
#u_petsc.setArray(u_binary.getArray(True))


# split function
(disp_sol_real, disp_sol_imag, pres_sol_real, pres_sol_imag) = u.split()

plot(pres_sol_real, interactive=True)

# create output directory and files
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
    except OSError, e:
        if e.errno != 17: # dir already exists (occurs due to MPI)
            raise
        pass
        
file_disp_real = File(output_dir+"/disp_real.pvd")
file_disp_imag = File(output_dir+"/disp_imag.pvd")
file_pres_real = File(output_dir+"/pres_real.pvd")
file_pres_imag = File(output_dir+"/pres_imag.pvd")

# Output solution
file_disp_real << disp_sol_real
file_disp_imag << disp_sol_imag
file_pres_real << pres_sol_real
file_pres_imag << pres_sol_imag
'''
