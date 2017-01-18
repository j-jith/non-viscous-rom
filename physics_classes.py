#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

from dolfin import *
parameters["reorder_dofs_serial"] = False

import numpy as np
from math import sqrt

class Grid(object):

    def __init__(self, mesh_filename):
        self.mesh = None
        self.subdomains = None 
        self.boundaries = None
        
        self.read_mesh(mesh_filename)

    def read_mesh(self, mesh_filename):
        print('Reading mesh...')
        # Read mesh
        self.mesh = Mesh(mesh_filename+".xml")
        self.boundaries = MeshFunction("size_t", self.mesh, mesh_filename+"_facet_region.xml")
        self.subdomains = MeshFunction("size_t", self.mesh, mesh_filename+"_physical_region.xml")

    def get_boundary_mesh(self):

        # get exterior boundary mesh
        bmesh = BoundaryMesh(self.mesh, 'exterior')
        # dimension of boundary mesh
        bdim = bmesh.topology().dim()
        # create meshfunction to mark boundaries on the boundary mesh
        b_boundaries = MeshFunction('size_t', bmesh, bdim)
        b_boundaries.set_all(0)

        for i, facet in enumerate(entities(bmesh, bdim)):
            parent_meshentity = bmesh.entity_map(bdim)[i]
            parent_boundarynumber = self.boundaries.array()[parent_meshentity]
            b_boundaries.array()[i] = parent_boundarynumber

        return bmesh, b_boundaries

    def get_mesh_volume(self):
        dx = Measure("dx", domain=self.mesh, subdomain_data=self.subdomains)
        return assemble(Constant(1.)*dx)

    def get_facet_normal(self, submesh, orientation='forward'):

        # 1D boundary mesh
        if submesh.type().dim() == 1:

            vertices = submesh.coordinates()
            cells = submesh.cells()

            vecs = vertices[cells[:, 1]] - vertices[cells[:, 0]]

            rot_mat = np.array([[0., -1.], [1., 0.]])

            vecs = np.dot(rot_mat, vecs.T).T
            vecs /= np.sqrt((vecs**2).sum(axis=1))[:, np.newaxis]

            #submesh.init_cell_orientations(Expression(('x[0]', 'x[1]'), degree=1))
            #vecs[submesh.cell_orientations() == 1] *= -1

            V = VectorFunctionSpace(submesh, 'DG', 0)
            normal = Function(V)

            for n in (0,1):
                dofmap = V.sub(n).dofmap()
                for i in xrange(dofmap.global_dimension()):
                    dof_indices = dofmap.cell_dofs(i)
                    assert len(dof_indices) == 1
                    normal.vector()[dof_indices[0]] = vecs[i, n]

        # 2D boundary mesh
        else:
            vertices = submesh.coordinates()
            cells = submesh.cells()

            vec1 = vertices[cells[:, 1]] - vertices[cells[:, 0]]
            vec2 = vertices[cells[:, 2]] - vertices[cells[:, 0]]

            nvecs = np.cross(vec1, vec2)
            nvecs /= np.sqrt((nvecs**2).sum(axis=1))[:, np.newaxis]

            # Ensure outward pointing normal
            submesh.init_cell_orientations(Expression(('x[0]', 'x[1]', 'x[2]'), degree=1))
            if orientation == 'forward':
                nvecs[submesh.cell_orientations() == 1] *= -1
            else:
                nvecs[submesh.cell_orientations() != 1] *= -1

            V = VectorFunctionSpace(submesh, 'DG', 0)
            normal = Function(V)

            for n in (0,1,2):
                dofmap = V.sub(n).dofmap()
                for i in xrange(dofmap.global_dimension()):
                    dof_indices = dofmap.cell_dofs(i)
                    assert len(dof_indices) == 1
                    normal.vector()[dof_indices[0]] = nvecs[i, n]

        return normal

class PhysicsBase(object):

    def __init__(self, meshfile):
        # mesh
        self.grid = Grid(meshfile)
        # Matrices
        self.K = PETScMatrix()
        self.M = PETScMatrix()
        self.f = PETScVector()
        
        self.ffc_params = {"optimize": True, \
                           "eliminate_zeros": True, \
                           "precompute_basis_const": True, \
                           "precompute_ip_const": True}

        self.element_type = 'CG'

        self.dofs_per_node = 1


    def linear_evp(self, **kwargs):

        eigensolver = SLEPcEigenSolver(self.K, self.M)

        if 'solver_params' in kwargs:
            for key in kwargs['solver_params']:
                eigensolver.parameters[key] = kwargs['solver_params'][key]

        if 'k' in kwargs:
            k = kwargs['k']
        else:
            k = 5

        eigensolver.solve(k)

        converged = eigensolver.get_number_converged()
        print("No. of converged eigenvalues: {}".format(converged))

        # no. of eigenpairs to return
        n_ret = (k if k<converged else converged)

        # get eigenvalues
        r = np.zeros((n_ret, ))
        c = np.zeros((n_ret, ))
        for i in range(n_ret):
            r[i], c[i] = eigensolver.get_eigenvalue(i)

        # sort eigenvalues (ascending order of magnitude)
        ii_sort = np.argsort(r**2+c**2)
        r = r[ii_sort]
        c = c[ii_sort]

        # get eigenvectors
        rx = []; cx = []
        for i in ii_sort:
            _, _, rx_i, cx_i = eigensolver.get_eigenpair(i)
            rx.append(rx_i)
            cx.append(cx_i)

        #r = []; c = []; rx = []; cx = []
        #for i in range(k if k<converged else converged):
        #    r_i, c_i, rx_i, cx_i = eigensolver.get_eigenpair(i)
        #    r.append(r_i)
        #    c.append(c_i)
        #    rx.append(rx_i)
        #    cx.append(cx_i)

        return r, c, rx, cx



class LinearSolid(PhysicsBase):

    def __init__(self, meshfile, degree, dirichlet_bc, props, **kwargs):
        super(LinearSolid, self).__init__(meshfile)

        self.degree = degree
        self.dimension = self.grid.mesh.geometry().dim()
        self.dirichlet_bc = dirichlet_bc

        self.dofs_per_node = self.dimension

        self.E = props['E']
        self.rho = props['rho']
        self.nu = props['nu']

        # Lame's constants
        self.mu_s = Constant(self.E/2./(1. + self.nu))
        self.lambda_s = Constant(self.E*self.nu/(1. + self.nu)/(1. - 2.*self.nu))

        # Check dimension 
        if self.dimension != 2 and self.dimension != 3:
            sys.exit('Specified spatial dimension is not 2 or 3.')

        # Parse kwargs
        if 'body_force' in kwargs:
            if kwargs['body_force']:
                self.body_force = kwargs['body_force']
        else:
            if self.dimension == 2:
                self.body_force = Constant((0., 0.))
            else:
                self.body_force = Constant((0., 0., 0.))

        if 'natural_bc' in kwargs:
            self.natural_bc = kwargs['natural_bc']
        else:
            self.natural_bc = {}

        if 'ffc_params' in kwargs:
            self.ffc_params = kwargs['ffc_params']

    # Stress tensor
    def sigma(self, v):
        return Constant(2.)*self.mu_s*sym(grad(v)) + self.lambda_s*tr(sym(grad(v)))*Identity(len(v))



    def assemble(self): 
        print('Assembling...')
        # Define measures and normal
        dx = Measure("dx", domain=self.grid.mesh, subdomain_data=self.grid.subdomains)
        ds = Measure("ds", domain=self.grid.mesh, subdomain_data=self.grid.boundaries)
        normal_vector = FacetNormal(self.grid.mesh)

        # Define function space
        #func_space = VectorFunctionSpace(self.grid.mesh, self.element_type, self.degree, dim=self.dimension)
        func_space = self.get_function_space()

        # Define test and trial functions
        u = TrialFunction(func_space)
        v = TestFunction(func_space)

        # Define solid material properties

        # Mass weak form
        #mass = Constant(-self.rho)*dot(v, u)*dx
        mass = Constant(self.rho)*dot(v, u)*dx

        # Stress tensor
        #sigma = 2*mu_s*sym(grad(u)) + lambda_s*tr(grad(u))*Identity(v.ufl_domain().geometric_dimension())

        # Stiffness weak form
        #stiff = inner(self.sigma(u), sym(grad(v)))*dx
        stiff = inner(sym(grad(v)), self.sigma(u))*dx

        # Dirichlet boundary condition
        solid_bcs = []
        if len(self.dirichlet_bc['values']) > 0:
            for value, boundary in zip(self.dirichlet_bc['values'], self.dirichlet_bc['boundaries']):
                solid_bcs.append(DirichletBC(func_space, Constant(value), self.grid.boundaries, boundary))
        else:
            sys.exit("Please set Dirichlet BC for solid.")

        # Body force
        rhs = dot(self.body_force, v)*dx

        # Natural boundary condition
        if len(self.natural_bc['values']) > 0:
            for value, boundary in zip(self.natural_bc['values'], self.natural_bc['boundaries']):
                rhs += dot(Constant(value), v)*ds(boundary)

        # # Assembly
        # assemble(mass, keep_diagonal=True, tensor=self.M, form_compiler_parameters=self.ffc_params)
        # assemble(stiff, keep_diagonal=True, tensor=self.K, form_compiler_parameters=self.ffc_params)
        # assemble(rhs, tensor=self.f, form_compiler_parameters=self.ffc_params)

        # # Apply boundary conditions
        # for bc in solid_bcs:
        #     bc.apply(self.M, self.f)
        #     bc.apply(self.K)
        #     #bc.zero(self.K)
        #     #bc.apply(self.K, self.f)
        #     #bc.zero(self.M)

        # Symmetric assembly
        assemble_system(mass, rhs, solid_bcs, A_tensor=self.M, b_tensor=self.f, form_compiler_parameters=self.ffc_params)
        assemble_system(stiff, rhs, solid_bcs, A_tensor=self.K, b_tensor=self.f, form_compiler_parameters=self.ffc_params)

    def get_function_space(self, **kwargs):
        if 'mesh' in kwargs:
            mesh = kwargs['mesh']
        else:
            mesh = self.grid.mesh

        return VectorFunctionSpace(mesh, self.element_type, self.degree, dim=self.dimension)

class AcousticFluid(PhysicsBase):

    def __init__(self, meshfile, degree, props, **kwargs):
        super(AcousticFluid, self).__init__(meshfile)

        self.degree = degree
        self.dimension = self.grid.mesh.geometry().dim()

        # Material properties
        self.c = props['c']
        self.rho = props['rho']


        # Check dimension 
        if self.dimension != 2 and self.dimension != 3:
            sys.exit('Specified spatial dimension is not 2 or 3.')

        # Parse kwargs
        if 'body_force' in kwargs:
            self.body_force = kwargs['body_force']
        else:
            self.body_force = 0.

        if 'dirichlet_bc' in kwargs:
            self.dirichlet_bc = kwargs['dirichlet_bc']
        else:
            self.dirichlet_bc = {}

        if 'natural_bc' in kwargs:
            self.natural_bc = kwargs['natural_bc']
        else:
            self.natural_bc = {}

        if 'ffc_params' in kwargs:
            self.ffc_params = kwargs['ffc_params']


    def assemble(self): 
        print('Assembling...')
        # Define measures and normal
        dx = Measure("dx", domain=self.grid.mesh, subdomain_data=self.grid.subdomains)
        ds = Measure("ds", domain=self.grid.mesh, subdomain_data=self.grid.boundaries)
        normal_vector = FacetNormal(self.grid.mesh)

        # Define function space
        #func_space = FunctionSpace(self.grid.mesh, self.element_type, self.degree)
        func_space = self.get_function_space()

        # Define test and trial functions
        p = TrialFunction(func_space)
        q = TestFunction(func_space)

        # Mass weak form
        mass = q*p*dx

        # Stiffness weak form
        stiff = Constant(self.c**2)*dot(grad(q), grad(p))*dx

        # Dirichlet boundary condition
        fluid_bcs = []
        if self.dirichlet_bc:
            for value, boundary in zip(self.dirichlet_bc['values'], self.dirichlet_bc['boundaries']):
                fluid_bcs.append(DirichletBC(func_space, Constant(value), self.grid.boundaries, boundary))

        # Body force
        rhs = Constant(self.body_force)*q*dx


        #assemble(mass, keep_diagonal=True, tensor=self.M, form_compiler_parameters=self.ffc_params)
        #assemble(stiff, keep_diagonal=True, tensor=self.K, form_compiler_parameters=self.ffc_params)
        #assemble(rhs, tensor=self.f, form_compiler_parameters=self.ffc_params)

        #M.ident_zeros()
        # Apply boundary conditions
        #for bc in fluid_bcs:
        #    bc.apply(self.M, self.f)
        #    bc.apply(self.Dv)
        #    bc.apply(self.Dh)
        #    bc.apply(self.K)

        assemble_system(mass, rhs, fluid_bcs, A_tensor=self.M, b_tensor=self.f, form_compiler_parameters=self.ffc_params)
        assemble_system(stiff, rhs, fluid_bcs, A_tensor=self.K, b_tensor=self.f, form_compiler_parameters=self.ffc_params)

    def get_function_space(self, **kwargs):
        if 'mesh' in kwargs:
            mesh = kwargs['mesh']
        else:
            mesh = self.grid.mesh

        return FunctionSpace(mesh, self.element_type, self.degree)


class ViscoThermalFluid(PhysicsBase):

    def __init__(self, meshfile, degree, props, **kwargs):
        super(ViscoThermalFluid, self).__init__(meshfile)

        self.degree = degree
        self.dimension = self.grid.mesh.geometry().dim()

        # Material properties
        self.c = props['c']
        self.rho = props['rho']
        self.cp = props['cp']
        self.gamma = props['gamma']
        self.kappa = props['kappa']
        self.mu = props['mu']
        self.lmbda = props['lmbda']

        # Viscothermal lengths
        self.lv = (2*self.mu + self.lmbda)/self.rho/self.c
        self.lvdash = self.mu/self.rho/self.c
        self.lh = self.kappa/self.rho/self.cp/self.c
        self.lvh = self.lv + (self.gamma-1)*self.lh


        # Check dimension 
        if self.dimension != 2 and self.dimension != 3:
            sys.exit('Specified spatial dimension is not 2 or 3.')

        # Parse kwargs
        if 'body_force' in kwargs:
            self.body_force = kwargs['body_force']
        else:
            self.body_force = 0.

        if 'dirichlet_bc' in kwargs:
            self.dirichlet_bc = kwargs['dirichlet_bc']
        else:
            self.dirichlet_bc = {}

        if 'natural_bc' in kwargs:
            self.natural_bc = kwargs['natural_bc']
        else:
            self.natural_bc = {}

        if 'ffc_params' in kwargs:
            self.ffc_params = kwargs['ffc_params']

        # Matrices
        self.Dv = PETScMatrix()
        self.Dh = PETScMatrix()

    def get_alphas(self, omega):
        alpha_m = (omega*(1 - 1j/2*omega*self.lvh))**2
        alpha_v = -(1-1j)/sqrt(2)*sqrt(self.mu/self.rho)/sqrt(omega)
        alpha_h = (1-1j)/sqrt(2)*(self.gamma-1)/self.c**2 \
                *sqrt(self.kappa/self.rho/self.cp)*omega*sqrt(omega)

        return alpha_m, alpha_v, alpha_h

    def get_alphas_deriv(self, omega):

        alpha_m1 = 2 * omega*(1 - 1j/2*omega*self.lvh) * (1 - 1j*omega*self.lvh)

        alpha_v1 = -(1-1j)/sqrt(2)*sqrt(self.mu/self.rho) * (-1/2/omega**(3/2))

        alpha_h1 = (1-1j)/sqrt(2)*(self.gamma-1)/self.c**2 \
                *sqrt(self.kappa/self.rho/self.cp) * (3/2*sqrt(omega))

        return alpha_m1, alpha_v1, alpha_h1


    def surf_lap_approx(self, test_func, trial_func, n_vec):

        surf_grad_test = grad(test_func) - dot(n_vec, grad(test_func))*n_vec
        surf_grad_trial = grad(trial_func) - dot(n_vec, grad(trial_func))*n_vec

        return -dot(surf_grad_test, surf_grad_trial)

    def assemble(self): 
        print('Assembling...')
        # Define measures and normal
        dx = Measure("dx", domain=self.grid.mesh, subdomain_data=self.grid.subdomains)
        ds = Measure("ds", domain=self.grid.mesh, subdomain_data=self.grid.boundaries)
        normal_vector = FacetNormal(self.grid.mesh)

        # Define function space
        #func_space = FunctionSpace(self.grid.mesh, self.element_type, self.degree)
        func_space = self.get_function_space()

        # Define test and trial functions
        p = TrialFunction(func_space)
        q = TestFunction(func_space)

        # Mass weak form
        mass = Constant(1/self.c**2)*q*p*dx

        # Stiffness weak form
        stiff = dot(grad(q), grad(p))*dx

        # Dirichlet boundary condition
        fluid_bcs = []
        if self.dirichlet_bc:
            for value, boundary in zip(self.dirichlet_bc['values'], self.dirichlet_bc['boundaries']):
                fluid_bcs.append(DirichletBC(func_space, Constant(value), self.grid.boundaries, boundary))

        # Body force
        rhs = Constant(self.body_force)*q*dx


        # Boundary layer impedance

        imp_v = self.surf_lap_approx(q, p, normal_vector)*ds
        imp_h = q*p*ds

        ## check signs of constants
        #alpha_v = -1/sqrt(2)*sqrt(mu/rho_0)
        #alpha_h = 1/sqrt(2)*(gamma-1)/c_0**2*sqrt(kappa/rho_0/c_p)

        #assemble(mass, keep_diagonal=True, tensor=self.M, form_compiler_parameters=self.ffc_params)
        #assemble(stiff, keep_diagonal=True, tensor=self.K, form_compiler_parameters=self.ffc_params)
        #assemble(imp_v, keep_diagonal=True, tensor=self.Dv, form_compiler_parameters=self.ffc_params)
        #assemble(imp_h, keep_diagonal=True, tensor=self.Dh, form_compiler_parameters=self.ffc_params)
        #assemble(rhs, tensor=self.f, form_compiler_parameters=self.ffc_params)

        #M.ident_zeros()
        # Apply boundary conditions
        #for bc in fluid_bcs:
        #    bc.apply(self.M, self.f)
        #    bc.apply(self.Dv)
        #    bc.apply(self.Dh)
        #    bc.apply(self.K)

        assemble_system(mass, rhs, fluid_bcs, A_tensor=self.M, b_tensor=self.f, form_compiler_parameters=self.ffc_params)
        assemble_system(stiff, rhs, fluid_bcs, A_tensor=self.K, b_tensor=self.f, form_compiler_parameters=self.ffc_params)

    def get_function_space(self, **kwargs):
        if 'mesh' in kwargs:
            mesh = kwargs['mesh']
        else:
            mesh = self.grid.mesh

        return FunctionSpace(mesh, self.element_type, self.degree)
