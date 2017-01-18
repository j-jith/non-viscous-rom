# Parameter file
from dolfin import Expression, Point

## Frequency sweep
### Initial frequency
f_0 = 150
### Perform sweep ?
sweep_flag = True
### Final frequency
f_1 = 950
### Number of steps
n_f = 801

## Mesh
grid_solid = "impeller_3d/structured/solid"

## Output directory for VTK files
output_full_flag = False
output_dir       = "output"

## Output point data file
point_data_flag = False
output_pfile    = "point_data_3d_aco"
output_point    = (0.1245, 0.0, 0.0135)
output_index    = 2 # z-displacement

## Degree of shape functions
degree_solid = 2

## Solid
### Young's modulus
E_s   = 196.5e9
### Poisson's ratio
nu_s  = 0.285
### Density of solid material
rho_s = 7989.

## Precribed displacement (Mandatory!)
### Boundary IDs at which disp. is prescribed (at least one boundary)
solid_dirichlet_boundary_id = [2]
### Corresponding prescribed disp.
solid_dirichlet_values      = [(0., 0., 0.)]

### Apply boundary load?
solid_boundary_load_flag = False
### Boundary IDs on which load is applied
solid_load_boundary_id   = []
### Corresponding boundary load values (as tuples)
solid_load_values        = []

class MyLoad(Expression):
    def eval(self, value, x):
        value[0] = 0
        value[1] = 0
        #value[2] = -1000

        centre = Point(0.120+0.053, 0, 0.014)

        if (x[0]-centre[0])**2 + (x[1]-centre[1])**2 + (x[2]-centre[2])**2 < 0.002**2:
            value[2] = -1000
        else:
            value[2] = 0

    def value_shape(self):
        return (3,)

#my_load = Expression(('0','0','1e6*exp(-((x[0]-0.15)*(x[0]-0.15) + x[1]*x[1])/1e-4)'), degree=2)
#my_load = Expression(('0.','0.','(x[0]-0.15)*(x[0]-0.15) + x[1]*x[1] < 0.01*0.01 ? 1e6 : 0.'), degree=1)
#my_load = (0,0,1000)
#my_load = MyLoad()

### Apply body force?
solid_body_force_flag = True
solid_body_force = MyLoad(degree=degree_solid)
