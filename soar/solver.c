#include <math.h>
#include <stdlib.h>
#include <petscmat.h>
#include <petscksp.h>

void read_mat_file(MPI_Comm comm, const char filename[], Mat *A)
{
    PetscInt n_rows = 0;
    PetscInt n_cols = 0;

    PetscPrintf(comm, "Reading matrix from %s ...\n", filename);

    PetscViewer viewer;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &viewer);

    MatCreate(comm, A);
    MatSetFromOptions(*A);
    MatLoad(*A, viewer);

    PetscViewerDestroy(&viewer);

    MatGetSize(*A, &n_rows, &n_cols);
    PetscPrintf(comm, "Shape: (%d, %d)\n", n_rows, n_cols);
}

void read_vec_file(MPI_Comm comm, const char filename[], Vec *b)
{
    PetscInt n_rows = 0;

    PetscPrintf(comm, "Reading vector from %s ...\n", filename);

    PetscViewer viewer;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &viewer);

    VecCreate(comm, b);
    VecSetFromOptions(*b);
    VecLoad(*b, viewer);

    PetscViewerDestroy(&viewer);

    VecGetSize(*b, &n_rows);
    PetscPrintf(comm, "Shape: (%d, )\n", n_rows);
}

void write_vec_file(MPI_Comm comm, const char filename[], Vec *b)
{
    PetscInt n_rows = 0;
    PetscPrintf(comm, "Writing vector to %s ...\n", filename);

    VecGetSize(*b, &n_rows);
    PetscPrintf(comm, "Shape: (%d, )\n", n_rows);

    PetscViewer viewer;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);
    
    VecView(*b, viewer);

    PetscViewerDestroy(&viewer);
}

void direct_solve(MPI_Comm comm, Mat *A, Vec *b, Vec *u)
{
    // Solve A * u = b

    KSP ksp; PC pc;

    KSPCreate(comm, &ksp);
    KSPSetOperators(ksp, *A, *A);
    KSPSetType(ksp, KSPPREONLY);

    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);

    PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);

    KSPSetFromOptions(ksp);

    PetscPrintf(PETSC_COMM_WORLD, "Solving ...\n");

    KSPSolve(ksp, *b, *u);

}

int main(int argc, char **args)
{
    // Fluid properties
    // Density of fluid material
    PetscReal rho_0 = 688.34;
    // Speed of sound in fluid
    PetscReal c_0   = 376.05;
    // Isobaric heat capacity of fluid
    PetscReal c_p    = 2739.8;
    // Ratio of heat capacities of fluid
    PetscReal gamma = 2.9408;
    // Thermal conductivity of fluid
    PetscReal kappa = 0.074537;
    // Viscosity of fluid
    PetscReal mu    = 5.5483e-5;
    // Damping coefficients
    PetscReal alpha_v = -1/sqrt(2)*sqrt(mu/rho_0);
    PetscReal alpha_h = 1/sqrt(2)*(gamma-1)/c_0/c_0*sqrt(kappa/rho_0/c_p);

    PetscReal omega = 2*M_PI*150;

    PetscReal omega2 = omega*omega;
    PetscReal omega_v = alpha_v/sqrt(omega);
    PetscReal omega_h = alpha_h*omega*sqrt(omega);

    Mat M, K, Dv, Dh, A; // Matrices
    Vec b, u; // Vectors


    char mass_file[] = "mass_matrix.dat";
    char stiff_file[] = "stiffness_matrix.dat";
    char damp_v_file[] = "damping_v_matrix.dat";
    char damp_h_file[] = "damping_h_matrix.dat";
    char load_file[] = "load_vector.dat";
    char sol_file[] = "solution_vector.dat";

    //PetscMPIInt rank, size;

    PetscInitialize(&argc, &args, NULL, NULL);
    //MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    //MPI_Comm_size(PETSC_COMM_WORLD,&size);

    if(argc > 1)
    {
        omega = 2*M_PI*atof(args[1]);
        omega2 = omega*omega;
        omega_v = alpha_v/sqrt(omega);
        omega_h = alpha_h*omega*sqrt(omega);
    }
    PetscPrintf(PETSC_COMM_WORLD, "Solving for frequency %f Hz.\n", atof(args[1]));

    read_mat_file(MPI_COMM_WORLD, mass_file, &M);
    read_mat_file(MPI_COMM_WORLD, stiff_file, &K);
    read_mat_file(MPI_COMM_WORLD, damp_v_file, &Dv);
    read_mat_file(MPI_COMM_WORLD, damp_h_file, &Dh);
    read_vec_file(MPI_COMM_WORLD, load_file, &b);


    PetscPrintf(PETSC_COMM_WORLD, "Adding up LHS matrices ...\n");
    // Add up matrices to A
    MatDuplicate(K, MAT_COPY_VALUES, &A);
    MatAXPY(A, omega2, M, DIFFERENT_NONZERO_PATTERN);
    MatAXPY(A, omega_v, Dv, DIFFERENT_NONZERO_PATTERN);
    MatAXPY(A, omega_h, Dh, DIFFERENT_NONZERO_PATTERN);

    // Create a right vector to store the solution
    MatCreateVecs(A, &u, NULL);
    // Solve
    direct_solve(MPI_COMM_WORLD, &A, &b, &u);

    // save solution
    write_vec_file(MPI_COMM_WORLD, sol_file, &u);

    // Free work space
    MatDestroy(&M);
    MatDestroy(&K);
    MatDestroy(&Dv);
    MatDestroy(&Dh);
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&u);

    PetscFinalize();

    return 0;
}
