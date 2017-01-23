#include "globals.h"

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

void get_block_diag_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1,
        PetscBool flip_signs)
{
    PetscInt i;
    PetscScalar sign=1.;

    if(flip_signs)
        sign = -1.;

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
        (*aj1)[i] = sign * aj[i];
        (*aj1)[i+ai[nrows]] = aj[i]+nrows;

        (*av1)[i] = sign * av[i];
        (*av1)[i+ai[nrows]] = av[i];
    }

}

void get_block_mass_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1)
{
    get_block_diag_csr(comm, ai, aj, av, nrows, ai1, aj1, av1, PETSC_TRUE);
}

void get_block_stiffness_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1)
{
    get_block_diag_csr(comm, ai, aj, av, nrows, ai1, aj1, av1, PETSC_FALSE);
}

void get_block_damping_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1,
        PetscInt **ai2, PetscInt **aj2, PetscScalar **av2)
{
    PetscInt i;

    PetscMalloc1(2*nrows+1, ai1);

    PetscMalloc1(2*nrows+1, ai2);

    for(i=0; i<nrows; i++)
    {
        (*ai1)[i] = ai[i];
        (*ai1)[i+nrows] = ai[i]+ai[nrows];

        (*ai2)[i] = ai[i];
        (*ai2)[i+nrows] = ai[i]+ai[nrows];
    }

    (*ai1)[2*nrows] = 2*ai[nrows];

    (*ai2)[2*nrows] = 2*ai[nrows];

    PetscMalloc1(ai[nrows]*2, aj1);
    PetscMalloc1(ai[nrows]*2, av1);

    PetscMalloc1(ai[nrows]*2, aj2);
    PetscMalloc1(ai[nrows]*2, av2);

    for(i=0; i<ai[nrows]; i++)
    {
        (*aj1)[i] = aj[i]+nrows;
        (*aj1)[i+ai[nrows]] = aj[i];

        (*av1)[i] = -av[i];
        (*av1)[i+ai[nrows]] = av[i];

        (*aj2)[i] = aj[i];
        (*aj2)[i+ai[nrows]] = aj[i]+nrows;

        (*av2)[i] = -av[i];
        (*av2)[i+ai[nrows]] = -av[i];
    }
}

/*
void get_block_damping_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscScalar g_real, PetscScalar g_imag,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1)
{
    PetscInt i, j, k;

    PetscMalloc1(2*nrows+1, ai1);

    for(i=0; i<nrows; i++)
    {
        (*ai1)[i] = 2*ai[i];
        (*ai1)[i+nrows] = 2*(ai[i]+ai[nrows]);
    }
    (*ai1)[2*nrows] = 4*ai[nrows];

    PetscMalloc1(4*ai[nrows], aj1);
    PetscMalloc1(4*ai[nrows], av1);

    for(i=0; i<nrows; i++)
    {
        k = ai[i+1]-ai[i];

        for(j=0; j<k; j++)
        {
            (*aj1)[i+j] = aj[i+j];
            (*aj1)[i+j+k] = aj[i+j]+nrows;
            (*aj1)[i+j+2*ai[nrows]] = aj[i+j];
            (*aj1)[i+j+k+2*ai[nrows]] = aj[i+j]+nrows;

            (*av1)[i+j] = -av[i+j] * g_imag;
            (*av1)[i+j+k] = -av[i+j] * g_real;
            (*av1)[i+j+2*ai[nrows]] = av[i+j] * g_real;
            (*av1)[i+j+k+2*ai[nrows]] = -av[i+j] * g_imag;
        }
    }

}
*/

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

void create_block_damping(MPI_Comm comm, Mat *M, Mat *M1, Mat *M2)
{
    PetscInt *ai, *aj, *ai1, *aj1, *ai2, *aj2;
    PetscScalar *av, *av1, *av2;
    PetscInt nrows;

    // Get CSR arrays of small matrix
    get_csr(comm, M, &ai, &aj, &av, &nrows);

    // Create CSR arrays of big matrix
    get_block_damping_csr(comm, ai, aj, av, nrows,
            &ai1, &aj1, &av1, &ai2, &aj2, &av2);

    // Create big matrix
    MatCreateMPIAIJWithArrays(comm, 2*nrows, 2*nrows, 2*nrows, 2*nrows,
            ai1, aj1, av1, M1);
    MatCreateMPIAIJWithArrays(comm, 2*nrows, 2*nrows, 2*nrows, 2*nrows,
            ai2, aj2, av2, M2);

    PetscFree(ai); PetscFree(aj); PetscFree(av);
    PetscFree(ai1); PetscFree(aj1); PetscFree(av1);
    PetscFree(ai2); PetscFree(aj2); PetscFree(av2);
}

/*
void create_block_damping(MPI_Comm comm, Mat *M,
        PetscScalar mu, PetscScalar omega, Mat *M1)
{
    PetscInt *ai, *aj, *ai1, *aj1;
    PetscScalar *av, *av1;
    PetscInt nrows;

    PetscScalar g_real, g_imag;

    g_real = (mu*mu) / (mu*mu + omega*omega);
    g_imag = (-mu*omega) / (mu*mu + omega*omega);

    // Get CSR arrays of small matrix
    get_csr(comm, M, &ai, &aj, &av, &nrows);

    // Create CSR arrays of big matrix
    get_block_damping_csr(comm, ai, aj, av, nrows, g_real, g_imag, &ai1, &aj1, &av1);

    // Create big matrix
    MatCreateMPIAIJWithArrays(comm, 2*nrows, 2*nrows, 2*nrows, 2*nrows,
            ai1, aj1, av1, M1);

    PetscFree(ai); PetscFree(aj); PetscFree(av);
    PetscFree(ai1); PetscFree(aj1); PetscFree(av1);
}
*/


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
