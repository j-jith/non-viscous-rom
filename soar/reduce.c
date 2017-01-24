#include "globals.h"

void project_matrix(MPI_Comm comm, Mat *M, Vec *Q, PetscInt n_q, Mat *A)
{
    PetscInt i, j;
    Vec tmp;
    PetscScalar val;

    MatCreate(comm, A);
    MatSetSizes(*A, n_q, n_q, PETSC_DETERMINE, PETSC_DETERMINE);
    MatSetType(*A, MATDENSE);
    MatSetUp(*A);

    MatCreateVecs(*M, NULL, &tmp);

    for(i=0; i<n_q; i++)
    {
        MatMult(*M, Q[i], tmp);
        for(j=0; j<n_q; j++)
        {
            VecDot(Q[j], Q[i], &val);
            MatSetValue(*A, i, j, val, INSERT_VALUES);
        }
    }

    MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
}

void project_vector(MPI_Comm comm, Vec *u, Vec *Q, PetscInt n_q, Vec *u_new)
{
    PetscInt i;
    PetscScalar val;

    VecCreate(comm, u_new);
    VecSetSizes(*u_new, n_q, n_q);
    VecSetUp(*u_new);

    for(i=0; i<n_q; i++)
    {
        VecDot(Q[i], *u, &val);
        VecSetValue(*u_new, i, val, INSERT_VALUES);
    }

    VecAssemblyBegin(*u_new);
    VecAssemblyEnd(*u_new);
}

void recover_vector(MPI_Comm comm, Vec *u, Vec *Q, PetscInt n_q, Vec *u_new)
{
    PetscInt i;
    PetscScalar *vals;

    VecDuplicate(Q[0], u_new);
    VecSet(*u_new, 0);

    VecGetArray(*u, &vals);

    for(i=0; i<n_q; i++)
    {
        VecAXPY(*u_new, vals[i], Q[i]);
    }

    VecRestoreArray(*u, &vals);
}

void recover_vectors(MPI_Comm comm, Vec *u, PetscInt n_u, Vec *Q, PetscInt n_q,
        Vec **u_new)
{
    PetscInt i, j;
    PetscScalar *vals;

    PetscMalloc1(n_u, u_new);

    for(j=0; j<n_u; j++)
    {
        VecDuplicate(Q[0], &(u_new[0][j]));
        VecSet(u_new[0][j], 0);

        VecGetArray(u[j], &vals);

        for(i=0; i<n_q; i++)
        {
            VecAXPY(u_new[0][j], vals[i], Q[i]);
        }

        VecRestoreArray(u[j], &vals);
    }
}

void direct_sweep_piecewise(MPI_Comm comm, Mat *M, Mat *C1, Mat *C2, Mat *K, Vec *b,
        PetscScalar omega_i, PetscScalar omega_f, PetscInt n_omega,
        Fitter *fits, PetscInt n_fits, Vec **u)
{
    PetscScalar *omegas, alpha, omega2;
    PetscInt i, j;
    Mat A;
    KSP ksp; PC pc;

    KSPCreate(comm, &ksp);
    KSPSetType(ksp, KSPPREONLY);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);

    omegas = linspace(omega_i, omega_f, n_omega);

    PetscMalloc1(n_omega, u);

    for(i=0; i<n_omega; i++)
    {
        omega2 = omegas[i]*omegas[i];

        MatDuplicate(*K, MAT_COPY_VALUES, &A);
        MatAXPY(A, omega2, *M, DIFFERENT_NONZERO_PATTERN);

        for(j=0; j<n_fits; j++)
        {
            alpha = fits[j].weights[i];

            // mass contribution
            MatAXPY(A, omega2*alpha*fits[j].c1[2], *C1, DIFFERENT_NONZERO_PATTERN);
            MatAXPY(A, omega2*alpha*fits[j].c2[2], *C2, DIFFERENT_NONZERO_PATTERN);

            // damping contribution
            MatAXPY(A, omegas[i]*alpha*fits[j].c1[1], *C1,
                    DIFFERENT_NONZERO_PATTERN);
            MatAXPY(A, omegas[i]*alpha*fits[j].c2[1], *C2,
                    DIFFERENT_NONZERO_PATTERN);

            // stiffness contribution
            MatAXPY(A, alpha*fits[j].c1[0], *C1, DIFFERENT_NONZERO_PATTERN);
            MatAXPY(A, alpha*fits[j].c2[0], *C2, DIFFERENT_NONZERO_PATTERN);
        }

        MatCreateVecs(A, NULL, &(u[0][i]));
        KSPSetOperators(ksp, A, A);
        KSPSolve(ksp, *b, u[0][i]);

        MatDestroy(&A);
    }

    KSPDestroy(&ksp);

}

void direct_sweep(MPI_Comm comm, Mat *M, Mat *C1, Mat *C2, Mat *K, Vec *b,
        PetscScalar omega_i, PetscScalar omega_f, PetscInt n_omega,
        PetscScalar mu, Vec **u)
{
    PetscScalar *omegas, omega2, g_real, g_imag;
    PetscInt i;
    Mat A;
    KSP ksp; PC pc;

    KSPCreate(comm, &ksp);
    KSPSetType(ksp, KSPPREONLY);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);
    PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);

    omegas = linspace(omega_i, omega_f, n_omega);

    PetscMalloc1(n_omega, u);

    for(i=0; i<n_omega; i++)
    {
        omega2 = omegas[i]*omegas[i];
        g_real = (mu*mu) / (mu*mu + omegas[i]*omegas[i]);
        g_imag = (-mu*omegas[i]) / (mu*mu + omegas[i]*omegas[i]);

        MatDuplicate(*K, MAT_COPY_VALUES, &A);
        MatAXPY(A, omega2, *M, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(A, omegas[i]*g_real, *C1, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(A, omegas[i]*g_imag, *C2, DIFFERENT_NONZERO_PATTERN);

        MatCreateVecs(A, NULL, &(u[0][i]));
        KSPSetOperators(ksp, A, A);
        KSPSolve(ksp, *b, u[0][i]);

        MatDestroy(&A);
    }

    KSPDestroy(&ksp);

}

void direct_solve_dense(MPI_Comm comm, Mat *A, Vec *b, Vec *u)
{
    // Solve A * u = b

    KSP ksp; PC pc;

    KSPCreate(comm, &ksp);
    KSPSetOperators(ksp, *A, *A);
    KSPSetType(ksp, KSPPREONLY);

    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);

    KSPSetFromOptions(ksp);

    PetscPrintf(PETSC_COMM_WORLD, "Solving ...\n");

    KSPSolve(ksp, *b, *u);

    KSPDestroy(&ksp);

}

