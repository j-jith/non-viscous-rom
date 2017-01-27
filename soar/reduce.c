#include "globals.h"

void project_matrix(MPI_Comm comm, Mat *M, Vec *Q, PetscInt n_q, Mat *A)
{
    PetscInt i, j;
    Vec tmp;
    PetscScalar val;

    MatCreate(comm, A);
    //MatSetSizes(*A, n_q, n_q, PETSC_DETERMINE, PETSC_DETERMINE);
    MatSetSizes(*A, n_q, n_q, n_q, n_q);
    MatSetType(*A, MATDENSE);
    MatSetUp(*A);

    MatCreateVecs(*M, NULL, &tmp);

    for(i=0; i<n_q; i++)
    {
        MatMult(*M, Q[i], tmp);
        for(j=0; j<n_q; j++)
        {
            VecDot(Q[j], tmp, &val);
            MatSetValue(*A, j, i, val, INSERT_VALUES);
        }
    }

    MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
}

void check_projection(MPI_Comm comm, Vec u, Vec *Q, PetscInt n_q)
{
    PetscInt i, *inds;
    PetscScalar *vals, norm;
    Vec u_new, u_tmp;

    // Projection
    VecCreate(comm, &u_new);
    VecSetSizes(u_new, n_q, n_q);
    VecSetType(u_new, VECMPI);
    VecSetUp(u_new);
    VecSet(u_new, 0);

    PetscMalloc1(n_q, &vals);
    PetscMalloc1(n_q, &inds);
    for(i=0; i<n_q; i++) inds[i]=i;

    VecMDot(u, n_q, Q, vals);
    VecSetValues(u_new, n_q, inds, vals, INSERT_VALUES);

    PetscPrintf(comm, "(");
    for(i=0; i<n_q; i++)
        PetscPrintf(comm, "%e, ", vals[i]);
    PetscPrintf(comm, ")\n");
    PetscFree(vals); PetscFree(inds);

    VecAssemblyBegin(u_new);
    VecAssemblyEnd(u_new);

    // Recovery
    VecDuplicate(u, &u_tmp);
    VecSet(u_tmp, 0);

    VecGetArray(u_new, &vals);
    VecMAXPY(u_tmp, n_q, vals, Q);
    VecRestoreArray(u_new, &vals);

    // Check norm of difference
    VecNorm(Q[0], NORM_2, &norm);
    PetscPrintf(comm, "*** Norm of Q[0]: %e ***\n", norm);
    VecNorm(u, NORM_2, &norm);
    PetscPrintf(comm, "*** Norm of actual: %e ***\n", norm);
    VecNorm(u_tmp, NORM_2, &norm);
    PetscPrintf(comm, "*** Norm of recovered: %e ***\n", norm);
    VecAXPY(u_tmp, -1.0, u);
    VecNorm(u_tmp, NORM_2, &norm);
    PetscPrintf(comm, "*** Norm of difference: %e ***\n", norm);

}

void project_vector(MPI_Comm comm, Vec *u, Vec *Q, PetscInt n_q, Vec *u_new)
{
    PetscInt i;
    //PetscScalar val;
    PetscInt *inds;
    PetscScalar *vals;

    VecCreate(comm, u_new);
    VecSetSizes(*u_new, n_q, n_q);
    VecSetUp(*u_new);
    VecSet(*u_new, 0);

    PetscMalloc1(n_q, &vals);
    PetscMalloc1(n_q, &inds);
    for(i=0; i<n_q; i++) inds[i]=i;

    VecMDot(*u, n_q, Q, vals);
    VecSetValues(*u_new, n_q, inds, vals, INSERT_VALUES);

    /*
    for(i=0; i<n_q; i++)
    {
        VecDot(Q[i], *u, &val);
        VecSetValue(*u_new, i, val, INSERT_VALUES);
    }
    */

    VecAssemblyBegin(*u_new);
    VecAssemblyEnd(*u_new);
}

void recover_vector(MPI_Comm comm, Vec *u, Vec *Q, PetscInt n_q, Vec *u_new)
{
    //PetscInt i;
    PetscScalar *vals;

    VecDuplicate(Q[0], u_new);
    VecSet(*u_new, 0);

    VecGetArray(*u, &vals);

    VecMAXPY(*u_new, n_q, vals, Q);
    /*for(i=0; i<n_q; i++)
    {
        VecAXPY(*u_new, vals[i], Q[i]);
    }*/

    VecRestoreArray(*u, &vals);
}

void recover_vectors(MPI_Comm comm, Vec *u, PetscInt n_u, Vec *Q, PetscInt n_q,
        Vec **u_new)
{
    //PetscInt i, j;
    PetscInt j;
    PetscScalar *vals;

    PetscMalloc1(n_u, u_new);

    for(j=0; j<n_u; j++)
    {
        VecDuplicate(Q[0], &(u_new[0][j]));
        VecSet(u_new[0][j], 0);

        VecGetArray(u[j], &vals);

        VecMAXPY(u_new[0][j], n_q, vals, Q);
        /*
        for(i=0; i<n_q; i++)
        {
            VecAXPY(u_new[0][j], vals[i], Q[i]);
        }
        */

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

/*
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

        MatCreateVecs(A, &(u[0][i]), NULL);
        VecSet(u[0][i], 0);
        KSPSetOperators(ksp, A, A);
        KSPSolve(ksp, *b, u[0][i]);

        MatDestroy(&A);
    }

    KSPDestroy(&ksp);

}
*/

void direct_sweep(MPI_Comm comm, Mat *M, Mat *C1, Mat *C2, Mat *K, Vec *b,
        PetscScalar omega_i, PetscScalar omega_f, PetscInt n_omega,
        PetscScalar mu, Vec **u)
{
    PetscScalar *omegas, omega2, g_real, g_imag;
    PetscInt i;
    Mat A;

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

        MatCreateVecs(A, &(u[0][i]), NULL);
        VecSet(u[0][i], 0);

        MatLUFactor(A, 0, 0, 0);
        MatSolve(A, *b, u[0][i]);

        MatDestroy(&A);
    }

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

