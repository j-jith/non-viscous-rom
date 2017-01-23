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
    //MatSetType(*A, MATSEQAIJ);
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

void get_csr(MPI_Comm comm, Mat *M,
        PetscInt **ai, PetscInt **aj, PetscScalar **av,
        PetscInt *total_rows)
{
    PetscInt start, end, nrows, ncols;
    const PetscInt *cols;
    const PetscScalar *vals;
    PetscInt i;

    MatGetOwnershipRange(*M, &start, &end);
    nrows = end-start;
    //PetscPrintf(comm, "Local rows = %d ...\n", nrows);
    PetscMalloc(sizeof(PetscInt)*(nrows+1), ai);
    *total_rows = nrows;

    (*ai)[0] = 0;
    for(i = 0; i < nrows; i++)
    {
        MatGetRow(*M, i+start, &ncols, NULL, NULL);
        (*ai)[i+1] = (*ai)[i] + ncols;
        MatRestoreRow(*M, i+start, &ncols, NULL, NULL);
    }
    //PetscPrintf(PETSC_COMM_SELF, "%d\n", (*ai)[1000]);

    PetscMalloc(sizeof(PetscInt)*(*ai)[nrows], aj);
    PetscMalloc(sizeof(PetscScalar)*(*ai)[nrows], av);

    for(i = 0; i < nrows; i++)
    {
        MatGetRow(*M, i+start, &ncols, &cols, &vals);
        PetscMemcpy((*aj)+(*ai)[i], cols, sizeof(PetscInt)*ncols);
        PetscMemcpy((*av)+(*ai)[i], vals, sizeof(PetscScalar)*ncols);
        MatRestoreRow(*M, i+start, &ncols, &cols, &vals);
    }

}

void get_block_mass_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1)
{
    PetscInt i;

    PetscMalloc1(2*nrows+1, ai1);

    for(i=0; i<nrows; i++)
    {
        (*ai1)[i] = ai[i];
        (*ai1)[i+nrows] = ai[i]+ai[nrows];
    }
    (*ai1)[2*nrows] = 2*ai[nrows];

    PetscMalloc1(ai[nrows]*2, aj1);
    PetscMalloc1(ai[nrows]*2, av1);

    for(i=0; i<ai[nrows]; i++)
    {
        (*aj1)[i] = aj[i];
        (*aj1)[i+ai[nrows]] = aj[i]+nrows;

        (*av1)[i] = av[i];
        (*av1)[i+ai[nrows]] = av[i];
    }

}

void get_block_stiffness_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1)
{
    get_block_mass_csr(comm, ai, aj, av, nrows, ai1, aj1, av1);
}

void get_block_damping_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1)
{
    PetscInt i;

    PetscMalloc1(2*nrows+1, ai1);

    for(i=0; i<nrows; i++)
    {
        (*ai1)[i] = ai[i];
        (*ai1)[i+nrows] = ai[i]+ai[nrows];
    }
    (*ai1)[2*nrows] = 2*ai[nrows];

    PetscMalloc1(ai[nrows]*2, aj1);
    PetscMalloc1(ai[nrows]*2, av1);

    for(i=0; i<ai[nrows]; i++)
    {
        (*aj1)[i] = aj[i]+nrows;
        (*aj1)[i+ai[nrows]] = aj[i];

        (*av1)[i] = -av[i];
        (*av1)[i+ai[nrows]] = av[i];
    }
}

void create_block_mass(MPI_Comm comm, Mat *M, Mat *M1)
{
    PetscInt *ai, *aj, *ai1, *aj1;
    PetscScalar *av, *av1;
    PetscInt nrows;

    // Get CSR arrays of small matrix
    get_csr(comm, M, &ai, &aj, &av, &nrows);

    // Create CSR arrays of big matrix
    get_block_mass_csr(comm, ai, aj, av, nrows, &ai1, &aj1, &av1);

    // Create big matrix
    MatCreateMPIAIJWithArrays(comm, 2*nrows, 2*nrows, 2*nrows, 2*nrows,
            ai1, aj1, av1, M1);

    PetscFree(ai); PetscFree(aj); PetscFree(av);
    PetscFree(ai1); PetscFree(aj1); PetscFree(av1);
}

void create_block_stiffness(MPI_Comm comm, Mat *M, Mat *M1)
{
    create_block_mass(comm, M, M1);
}

void create_block_damping(MPI_Comm comm, Mat *M, Mat *M1)
{
    PetscInt *ai, *aj, *ai1, *aj1;
    PetscScalar *av, *av1;
    PetscInt nrows;

    // Get CSR arrays of small matrix
    get_csr(comm, M, &ai, &aj, &av, &nrows);

    // Create CSR arrays of big matrix
    get_block_damping_csr(comm, ai, aj, av, nrows, &ai1, &aj1, &av1);

    // Create big matrix
    MatCreateMPIAIJWithArrays(comm, 2*nrows, 2*nrows, 2*nrows, 2*nrows,
            ai1, aj1, av1, M1);

    PetscFree(ai); PetscFree(aj); PetscFree(av);
    PetscFree(ai1); PetscFree(aj1); PetscFree(av1);
}


void create_block_load(MPI_Comm comm, Vec *f, Vec *f1)
{
    PetscInt i, nrows;
    PetscScalar *v;

    VecGetSize(*f, &nrows);

    VecCreate(comm, f1);
    VecSetSizes(*f1, 2*nrows, 2*nrows);
    VecSetType(*f1, VECMPI);
    VecSet(*f1, 0);

    VecGetArray(*f, &v);

    for(i = 0;  i < nrows; i++)
    {
        VecSetValues(*f1, 1, &i, &v[i], INSERT_VALUES);
    }

    VecAssemblyBegin(*f1);
    VecAssemblyEnd(*f1);

    VecRestoreArray(*f, &v);
    //PetscFree(v);

}


int main(int argc, char **args)
{

    PetscReal f_0, f_1, omega;
    PetscInt n_f;


    Mat M0, K0, C0; // Small matrices
    Mat M, K, C, A; // Big matrices
    Vec b0; // Small vectors
    Vec b, u; // Big vectors

    // Loop counter
    PetscInt i = 0;


    PetscMPIInt rank, size;

    char mass_file[100] = "../matrices/mass.dat";
    char stiff_file[100] = "../matrices/stiffness.dat";
    char damp_file[100] = "../matrices/damping.dat";
    char load_file[100] = "../matrices/force.dat";
    char sol_file[100] = "../output/full/solution.dat";


    PetscInitialize(&argc, &args, NULL, NULL);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    if(argc > 3)
    {
        f_0 = atof(args[1]);
        f_1 = atof(args[2]);
        n_f = atoi(args[3]);
    }
    else return -1;


    // Read matrices from disk
    read_mat_file(PETSC_COMM_WORLD, mass_file, &M0);
    read_mat_file(PETSC_COMM_WORLD, stiff_file, &K0);
    read_mat_file(PETSC_COMM_WORLD, damp_file, &C0);
    read_vec_file(PETSC_COMM_WORLD, load_file, &b0);

    MatAssemblyBegin(M0, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(M0, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(K0, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(K0, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(C0, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(C0, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(b0); VecAssemblyEnd(b0);

    //PetscSequentialPhaseBegin(PETSC_COMM_SELF, 1);

    // Create block matrices
    create_block_mass(PETSC_COMM_WORLD, &M0, &M);
    create_block_stiffness(PETSC_COMM_WORLD, &K0, &K);
    create_block_damping(PETSC_COMM_WORLD, &C0, &C);
    create_block_load(PETSC_COMM_WORLD, &b0, &b);

    //PetscSequentialPhaseEnd(PETSC_COMM_SELF, 1);

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY); MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(b); VecAssemblyEnd(b);

    // Destroy small matrices
    MatDestroy(&M0);
    MatDestroy(&K0);
    MatDestroy(&C0);
    VecDestroy(&b0);


    // Create solver context
    KSP ksp; PC pc;

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetType(ksp, KSPPREONLY);

    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);

    PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);

    KSPSetFromOptions(ksp);

    // Create matrix for dynamic stiffness
    //MatDuplicate(K, MAT_DO_NOT_COPY_VALUES, &A);
    // Create a right vector to store the solution
    MatCreateVecs(K, &u, NULL);

    for(i=0; i<n_f; i++)
    {
        omega = (f_0 + (PetscReal) i*(f_1 - f_0)/n_f)*2*M_PI;
        PetscPrintf(PETSC_COMM_WORLD, "Solving for frequency %f Hz\n", omega/2/M_PI);

        PetscPrintf(PETSC_COMM_WORLD, "... Constructing dynamic stiffness\n");
        MatDuplicate(K, MAT_COPY_VALUES, &A);
        //MatCopy(K, A, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(A, -omega*omega, M, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(A, omega, C, DIFFERENT_NONZERO_PATTERN);

        PetscPrintf(PETSC_COMM_WORLD, "... Solving\n");
        VecSet(u, 0);
        KSPSetOperators(ksp, A, A);
        KSPSolve(ksp, b, u);
        MatDestroy(&A);

        // save solution
        sprintf(sol_file, "../output/full/solution_%i.dat", i);
        write_vec_file(MPI_COMM_WORLD, sol_file, &u);
    }


    // Free work space
    MatDestroy(&M);
    MatDestroy(&K);
    MatDestroy(&C);
    //MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&u);

    PetscFinalize();

    return 0;
}
