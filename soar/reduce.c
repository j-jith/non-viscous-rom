#include "globals.h"

void read_arnoldi_basis(MPI_Comm comm, const char dirname[], PetscInt *ind_ip, PetscInt len_ip, PetscInt n_arn, Vec *Q)
{
    char filename[100];
    PetscInt i, j;

    Q = (Vec*)malloc(sizeof(Vec)*len_ip*n_arn);

    for(i=0; i<len_ip; i++)
    {
        for(j=0; j<n_arn; j++)
        {
            sprintf(filename, "%s/basis_%d_%d.dat", dirname, ind_ip[i], j);
            read_vec_file(comm, filename, &(Q[i*len_ip+j]));
            strcpy(filename, "");
        }
    }
}

/*void generate_reduced_matrix(MPI_Comm comm, const char filename[], Vec *Q, PetscInt len_q, Mat *A)
{
    PetscInt i=0, j=0;
    Mat K;
    Vec *tmp_vec;
    PetscScalar *Aij;
    PetscInt *ind_i, *ind_j;

    PetscInt istart, iend;

    Aij = (PetscScalar*)malloc(sizeof(PetscScalar)*len_q*len_q);
    ind_i = (PetscInt*)malloc(sizeof(PetscInt)*len_q*len_q);
    ind_j = (PetscInt*)malloc(sizeof(PetscInt)*len_q*len_q);

    tmp_vec = (Vec*)malloc(sizeof(Vec)*len_q);

    read_mat_file(comm, filename, &K);

    PetscPrintf(comm, "Populating reduced matrix ...\n");

    PetscPrintf(comm, "DEBUG: For loop I.\n");
    for(i=0; i<len_q; i++)
    {
        MatCreateVecs(K, NULL, &(tmp_vec[i]));
        MatMult(K, Q[i], tmp_vec[i]);
    }

    PetscPrintf(comm, "DEBUG: For loop II.\n");
    for(i=0; i<len_q; i++)
    {
        for(j=0; j<len_q; j++)
        {
            VecDot(tmp_vec[j], Q[i], &(Aij[i*len_q+j]));
            ind_i[i*len_q+j] = i;
            ind_j[i*len_q+j] = j;
        }
    }

    PetscPrintf(comm, "DEBUG: Setting values.\n");
    //MatCreate(comm, A);
    //MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, len_q, len_q);
    //MatSetType(*A, MATDENSE);
    //MatSetFromOptions(*A);
    
    MatGetOwnershipRange(*A, &istart, &iend);
    for(i=istart; i<iend; i++)
    {
        PetscPrintf(comm, "DEBUG: Row %d.\n", i);
        MatSetValues(*A, len_q, &(ind_i[i*len_q]), len_q, &(ind_j[i*len_q]), &(Aij[i*len_q]), INSERT_VALUES);
    }
    MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);

    //MatSetValues(*A, len_q, ind_i, len_q, ind_j, Aij, INSERT_VALUES);
    //MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, len_q, len_q, NULL, A);
    //MatSetFromOptions(*A);

    MatDestroy(&K);
    //VecDestroy(&tmp_vec);
    free(Aij); free(ind_i); free(ind_j);

    for(i=0; i<len_q; i++)
    {
        VecDestroy(&(tmp_vec[j]));
    }
    free(tmp_vec);
}*/

/*void generate_reduced_vector(MPI_Comm comm, const char filename[], Vec *Q, PetscInt len_q, Vec *b)
{
    PetscInt i;
    Vec B;
    PetscScalar *b_i;
    PetscInt *ind_i;

    b_i = (PetscScalar*)malloc(sizeof(PetscScalar)*len_q);
    ind_i = (PetscInt*)malloc(sizeof(PetscInt)*len_q);

    read_vec_file(comm, filename, &B);

    PetscPrintf(comm, "Populating reduced vector ...\n");

    VecCreate(comm, b);
    VecSetSizes(*b, PETSC_DECIDE, len_q);
    VecSetFromOptions(*b);

    for(i=0; i<len_q; i++)
    {
            
        VecDot(Q[i], B, &(b_i[i]));
        //ind_i[i] = i;
    }

    //VecSetValues(*b, len_q, ind_i, b_i, INSERT_VALUES);
    VecCreateSeqWithArray(comm, 1, len_q, b_i, b);
    VecAssemblyBegin(*b);
    VecAssemblyEnd(*b);

    VecDestroy(&B);
    free(b_i); free(ind_i); 
}*/

void orthogonalize_arnoldi_disk(MPI_Comm comm, const char dirname[], PetscInt *ind_ip, PetscInt len_ip, PetscInt n_arn, Vec *Q, PetscInt *q_len)
{
    char filename[100];
    PetscInt i, j, k;
    Vec q_tmp;
    PetscScalar t=0; // dot product holder
    PetscReal norm=0; 

    PetscPrintf(comm, "Reading and orthogonalising union of Arnoldi basis");

    (*q_len) = 0;
    //Q = (Vec*)malloc(sizeof(Vec)*len_ip*n_arn);

    for(i=0; i<len_ip; i++)
    {
        for(j=0; j<n_arn; j++)
        {
            sprintf(filename, "%s/basis_%d_%d.dat", dirname, ind_ip[i], j);
            read_vec_file(comm, filename, &q_tmp);
            strcpy(filename, "");

            VecNorm(q_tmp, NORM_2, &norm);

            // DEBUG
            PetscPrintf(comm, "Norm = %le\n", norm);

            VecScale(q_tmp, 1/norm);

            for(k=0; k<(*q_len); k++)
            {
                VecDot(Q[k], q_tmp, &t);
                VecAXPY(q_tmp, -t, Q[k]);
            }

            VecNorm(q_tmp, NORM_2, &norm);
            if(norm > TOLERANCE)
            {
                VecScale(q_tmp, 1/norm);
                VecDuplicate(q_tmp, &(Q[(*q_len)]));
                VecCopy(q_tmp, Q[(*q_len)]);
                (*q_len)++;
            }
            VecDestroy(&q_tmp);

        }
    }

    PetscPrintf(comm, "Returning %d Arnoldi basis.\n", (*q_len));

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

}


void generate_reduced_matrix(MPI_Comm comm, const char filename[], Vec *Q, PetscInt len_q, const char outfile[])
{
    PetscInt i=0, j=0;
    Mat K;
    Vec *tmp_vec;
    PetscScalar **Aij;

    FILE *fp;


    Aij = (PetscScalar**)malloc(sizeof(PetscScalar*)*len_q);
    for(i=0; i<len_q; i++)
    {
        Aij[i] = (PetscScalar*)malloc(sizeof(PetscScalar)*len_q);
    }

    tmp_vec = (Vec*)malloc(sizeof(Vec)*len_q);

    read_mat_file(comm, filename, &K);

    PetscPrintf(comm, "Populating reduced matrix ...\n");

    PetscPrintf(comm, "DEBUG: For loop I.\n");
    for(i=0; i<len_q; i++)
    {
        MatCreateVecs(K, NULL, &(tmp_vec[i]));
        MatMult(K, Q[i], tmp_vec[i]);
    }

    PetscPrintf(comm, "DEBUG: For loop II.\n");
    for(i=0; i<len_q; i++)
    {
        PetscPrintf(comm, "Row %d.\n", i);
        for(j=0; j<len_q; j++)
        {
            VecDot(tmp_vec[j], Q[i], &(Aij[i][j]));
        }
    }

    // write to file
    fp = fopen(outfile, "w");
    for(i=0; i<len_q; i++)
    {
        for(j=0; j<len_q; j++)
        {
            fprintf(fp, "%le ", Aij[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);


    for(i=0; i<len_q; i++)
    {
        VecDestroy(&(tmp_vec[j]));
    }
    free(tmp_vec);
    MatDestroy(&K);
    //VecDestroy(&tmp_vec);
    free(Aij); 

}

void generate_reduced_vector(MPI_Comm comm, const char filename[], Vec *Q, PetscInt len_q, const char outfile[])
{
    PetscInt i;
    Vec B;
    PetscScalar *b_i;
    FILE *fp;

    b_i = (PetscScalar*)malloc(sizeof(PetscScalar)*len_q);

    read_vec_file(comm, filename, &B);

    PetscPrintf(comm, "Populating reduced vector ...\n");


    for(i=0; i<len_q; i++)
    {
            
        VecDot(Q[i], B, &(b_i[i]));
        //ind_i[i] = i;
    }

    // write to file
    fp = fopen(outfile, "w");
    for(i=0; i<len_q; i++)
    {
        fprintf(fp, "%le ", b_i[i]);
    }
    fclose(fp);


    VecDestroy(&B);
    free(b_i);
}
