#include "globals.h"

void get_covariance(MPI_Comm comm, Vec *Q, PetscInt n, Mat *R)
{
    PetscPrintf(comm, "Assembling covariance matrix... ");

    PetscInt i=0, j=0;
    PetscScalar val;

    MatCreate(comm, R);
    MatSetSizes(*R, n, n, PETSC_DETERMINE, PETSC_DETERMINE);
    MatSetType(*R, MATDENSE);
    MatSetUp(*R);

    for(i=0; i<n ; i++)
    {
        for(j=0; j<i; j++)
        {
            VecDot(Q[i], Q[j], &val);
            MatSetValue(*R, i, j, val, INSERT_VALUES);
            MatSetValue(*R, j, i, val, INSERT_VALUES);
        }
        VecDot(Q[i], Q[i], &val);
        MatSetValue(*R, i, i, val, INSERT_VALUES);
    }

    MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY);

    PetscPrintf(comm, "Done\n");

}

void get_pod_eigenvectors(MPI_Comm comm, Mat *A, PetscScalar tol,
        Vec **xr, PetscInt *rank)
{
    PetscPrintf(comm, "Computing POD eigenvectors... ");

    EPS eps;
    PetscInt nconv, i;
    PetscScalar kr=0, kr1=0;

    // Create eigenvalue solver context
    EPSCreate(comm, &eps);
    EPSSetType(eps, EPSLAPACK); // Use LAPACK
    EPSSetOperators(eps, *A, NULL);
    EPSSetProblemType(eps, EPS_HEP); // Hermitian
    EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL); // Sort by real part
    EPSSolve(eps);

    // Find number of converged eigenvalues
    EPSGetConverged(eps, &nconv);
    PetscPrintf(comm, "No. of converged eigenvalues: %D\n", nconv);

    // Create first eigenvector
    PetscMalloc1(nconv, xr);
    MatCreateVecs(*A, NULL, &(xr[0][0]));

    // Get first eigenpair
    EPSGetEigenpair(eps, 0, &kr1, NULL, xr[0][0], NULL);
    VecScale(xr[0][0], 1.0/sqrt(kr1)); // Scale eigenvector by sqrt of eigenvalue
    *rank = 1;

    PetscPrintf(comm, "%9f, ", (double)kr1);

    // Find all other eigenpairs
    i = 1;
    while( i < nconv)
    {
        EPSGetEigenvalue(eps, i, &kr, NULL);
        if(kr/kr1 > tol)
        {
            VecDuplicate(xr[0][0], &(xr[0][*rank]));
            EPSGetEigenvector(eps, i, xr[0][*rank], NULL);

            VecScale(xr[0][*rank], 1.0/sqrt(kr)); // Scale eigenvector by sqrt of evalue
            (*rank)++;

            PetscPrintf(comm, "%9f, ", (double)kr);
        }
        i++;
    }
    PetscPrintf(comm, "\n");

    PetscPrintf(comm, "Rank: %D (tol = %e)\n", *rank, tol);

    // Clean-up
    EPSDestroy(&eps);
   /* if( nconv > (*rank) )
    {
        for(i=(*rank); i<nconv; i++)
            VecDestroy(&(xr[i]));
    }*/

    PetscPrintf(comm, "Done\n");
}

void pod_orthogonalise(MPI_Comm comm, Vec *Q, PetscInt n_q, PetscScalar tol,
        Vec **Q1, PetscInt *rank)
{
    PetscPrintf(comm, "POD Orthogonalisation...\n");

    Mat R;
    Vec *xr=NULL;
    PetscInt i, j;
    PetscScalar *vals=NULL;
    //PetscScalar norm=0;

    // get covariance matrix of old basis vectors
    get_covariance(comm, Q, n_q, &R);
    // get the eigenvectors of covariance matrix
    // for POD orthogonalisation
    get_pod_eigenvectors(comm, &R, tol, &xr, rank);
    // Free the covariance matrix
    MatDestroy(&R);

    // Alloc space for new basis vectors
    PetscMalloc1(*rank, Q1);

    PetscPrintf(comm, "Transforming POD eigenvectors...\n");

    for(i=0; i<(*rank); i++)
    {
        // Create new basis vector
        VecDuplicate(Q[0], &(Q1[0][i]));
        VecSet(Q1[0][i], 0); // Initialise to zero

        // Get values of POD eigenvector
        VecGetArray(xr[i], &vals);

        // Compute new basis vector
        for(j=0; j<n_q; j++)
        {
            VecAXPY(Q1[0][i], vals[j], Q[j]);
        }

        // Restore POD eigenvector (required)
        VecRestoreArray(xr[i], &vals);
        // Free POD eigenvector
        VecDestroy(&(xr[i]));

        //VecNorm(Q1[0][i], NORM_2, &norm);
        //VecScale(Q1[0][i], 1.0/norm);
    }

    PetscFree(xr);

    PetscPrintf(comm, "Returning %D vectors...\n", *rank);

    PetscPrintf(comm, "Done\n");

}

void check_orthogonality(MPI_Comm comm, Vec *Q, PetscInt n_q)
{
    PetscInt i, j;
    PetscScalar val;

    for(i=0; i<n_q; i++)
    {
        PetscPrintf(comm, "Row %D: ", i);

        for(j=0; j<=i; j++)
        {
            VecDot(Q[i], Q[j], &val);
            PetscPrintf(comm, "(%D, %D) = %e, ", i, j, val);
        }
        PetscPrintf(comm, "\n");
    }
}
