#include "globals.h"

int main(int argc, char **args)
{
    char mass_file[] = "../matrices/mass.dat";
    char stiff_file[] = "../matrices/stiffness.dat";
    char damp_file[] = "../matrices/damping.dat";
    char load_file[] = "../matrices/force.dat";
    char sol_file[100];


    PetscReal omega_i, omega_f, g_real, g_imag, mu=825.056;
    PetscInt omega_len;
    PetscReal *omega;

    Mat M0, K0, C0; // Small matrices
    Mat M, K, C1, C2, A; // Big matrices
    Vec b0; // Small vectors
    Vec b, u; // Big vectors

    PetscInt i;

    PetscInitialize(&argc, &args, NULL, NULL);

    if(argc > 3)
    {
        omega_i = (PetscReal)atof(args[1])*2*M_PI;
        omega_f = (PetscReal)atof(args[2])*2*M_PI;
        omega_len = (PetscInt)atoi(args[3]);
    }
    else
    {
        //ind_ip = (PetscInt)round(omega_len/2);
        PetscPrintf(MPI_COMM_WORLD, "Usage: ./soar <initial freq.> <final freq.> <no. of freqs.>\n");
        return 0;
    }

    omega = linspace(omega_i, omega_f, omega_len);

    // Read matrices from disk
    read_mat_file(MPI_COMM_WORLD, mass_file, &M0);
    read_mat_file(MPI_COMM_WORLD, stiff_file, &K0);
    read_mat_file(MPI_COMM_WORLD, damp_file, &C0);
    read_vec_file(MPI_COMM_WORLD, load_file, &b0);

    // Create block matrices
    create_block_mass(PETSC_COMM_WORLD, &M0, &M);
    create_block_stiffness(PETSC_COMM_WORLD, &K0, &K);
    create_block_damping(PETSC_COMM_WORLD, &C0, &C1, &C2);
    create_block_load(PETSC_COMM_WORLD, &b0, &b);

    // Destroy small matrices
    MatDestroy(&M0); MatDestroy(&C0);
    MatDestroy(&K0); VecDestroy(&b0);

    // Read fit coeffs and weights from file
    //Fitter *fit_list  = read_fitter(fit_file, weights_file, &n_fits);

    // Create solver context
    KSP ksp; PC pc;

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetType(ksp, KSPPREONLY);

    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);

    PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);

    KSPSetFromOptions(ksp);

    MatCreateVecs(K, &u, NULL);

    for(i=0; i<omega_len; i++)
    {
        PetscPrintf(PETSC_COMM_WORLD, "Solving for frequency %f Hz\n", omega[i]/2/M_PI);


        PetscPrintf(PETSC_COMM_WORLD, "... Constructing non-vicous damping matrix\n");
        //create_block_damping(PETSC_COMM_WORLD, &C0, mu, omega, &C);
        g_real = (mu*mu) / (mu*mu + omega[i]*omega[i]);
        g_imag = (-mu*omega[i]) / (mu*mu + omega[i]*omega[i]);

        PetscPrintf(PETSC_COMM_WORLD, "... Constructing dynamic stiffness\n");
        MatDuplicate(K, MAT_COPY_VALUES, &A);
        MatAXPY(A, omega[i]*omega[i], M, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(A, g_real*omega[i], C1, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(A, g_imag*omega[i], C2, DIFFERENT_NONZERO_PATTERN);

        PetscPrintf(PETSC_COMM_WORLD, "... Solving\n");
        VecSet(u, 0);
        KSPSetOperators(ksp, A, A);
        KSPSolve(ksp, b, u);

        // In-loop clean-up
        MatDestroy(&A);

        // save solution
        sprintf(sol_file, "output/full/solution/%i.dat", i);
        write_vec_file(MPI_COMM_WORLD, sol_file, &u);
    }

    // Free work space
    KSPDestroy(&ksp);
    MatDestroy(&M);
    MatDestroy(&K);
    MatDestroy(&C1);
    MatDestroy(&C2);
    VecDestroy(&b);
    VecDestroy(&u);

    PetscFinalize();

    return 0;
}

