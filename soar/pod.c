#include "globals.h"

void get_covariance(MPI_Comm comm, Vec *Q, PetscInt n, Mat *R)
{
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

}

void get_pod_eigenvectors(MPI_Comm comm, Mat *A, PetscScalar tol, Vec *xr, PetscInt *rank)
{
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
    PetscMalloc1(nconv, &xr);
    MatCreateVecs(*A, NULL, xr);

    // Get first eigenpair
    EPSGetEigenpair(eps, 0, &kr1, NULL, *xr, NULL);
    VecScale(*xr, 1.0/sqrt(kr1)); // Scale eigenvector by sqrt of eigenvalue
    *rank = 1;

    PetscPrintf(comm, " %9f \n", (double)kr1);

    // Find all other eigenpairs
    for(i=1; i<nconv; i++)
    {
        EPSGetEigenvalue(eps, i, &kr, NULL);

        // If the ratio of eigenvalue with the first eigenvalue
        // is less than POD tolerance, ignore it
        if(kr/kr1 > tol)
        {
            VecDuplicate(*xr, xr+i);
            EPSGetEigenvector(eps, i, xr[i], NULL);
            VecScale(xr[i], 1.0/sqrt(kr)); // Scale eigenvector by sqrt of evalue
            (*rank)++;

            PetscPrintf(comm, " %9f \n", (double)kr);
        }
        else break;
    }
    PetscPrintf(comm, "Rank: %D (tol = %e)\n", *rank, tol);

    // Clean-up
    EPSDestroy(&eps);
    if( nconv > (*rank) )
    {
        for(i=(*rank); i<nconv; i++)
            VecDestroy(xr+i);
    }
}

void pod_orthogonalise(MPI_Comm comm, Vec *Q, PetscInt n_q, PetscScalar tol,
        Vec *Q1, PetscInt *rank)
{
    Mat R;
    Vec *xr=NULL;
    PetscInt i, j;
    PetscScalar *vals;

    // get covariance matrix of old basis vectors
    get_covariance(comm, Q, n_q, &R);
    // get the eigenvectors of covariance matrix
    // for POD orthogonalisation
    get_pod_eigenvectors(comm, &R, tol, xr, rank);
    // Free the covariance matrix
    MatDestroy(&R);

    // Alloc space for new basis vectors
    PetscMalloc1(*rank, &Q1);


    for(i=0; i<(*rank); i++)
    {
        // Create new basis vector
        VecDuplicate(Q[0], Q1+i);
        VecSet(Q1[i], 0); // Initialise to zero

        // Get values of POD eigenvector
        VecGetArray(xr[i], &vals);

        // Compute new basis vector
        for(j=0; j<n_q; j++)
        {
            VecAXPY(Q1[i], vals[j], Q[j]);
        }

        // Restore POD eigenvector (required)
        VecRestoreArray(xr[i], &vals);
        // Free POD eigenvector
        VecDestroy(xr+i);
    }

}
