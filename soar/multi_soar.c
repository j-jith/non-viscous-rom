#include "globals.h"

PetscReal** generate_local_coeffs(MPI_Comm comm, Fitter *fits, PetscInt n_fits, PetscReal ip, PetscInt n_ip)
{
    PetscInt i;
    PetscReal damp_cc[6], **coeffs;
    PetscMalloc(sizeof(PetscReal*)*3, &coeffs);

    PetscPrintf(comm, "Generating local coeffs. for frequency %f Hz ...\n", ip/2/M_PI);

    // initialise coeffs
    for(i=0; i<6; i++)
    {
        damp_cc[i] = 0;
    }

    for(i=0; i<n_fits; i++)
    {
        damp_cc[0] += fits[i].c1[2]*fits[i].weights[n_ip]; // C1 contribution to M
        damp_cc[1] += fits[i].c2[2]*fits[i].weights[n_ip]; // C2 contribution to M

        damp_cc[2] += fits[i].c1[1]*fits[i].weights[n_ip]; // C1 contribution to D
        damp_cc[3] += fits[i].c2[1]*fits[i].weights[n_ip]; // C2 contribution to D

        damp_cc[4] += fits[i].c1[0]*fits[i].weights[n_ip]; // C1 contribution to K
        damp_cc[5] += fits[i].c2[0]*fits[i].weights[n_ip]; // C2 contribution to K
    }

    // mass
    PetscMalloc(sizeof(PetscReal)*2, &(coeffs[0]));
    //coeffs[0] = (PetscReal*)malloc(sizeof(PetscReal)*2);
    coeffs[0][0] = damp_cc[0];
    coeffs[0][1] = damp_cc[1];

    // damping
    PetscMalloc(sizeof(PetscReal)*3, &(coeffs[1]));
    //coeffs[1] = (PetscReal*)malloc(sizeof(PetscReal)*3);
    coeffs[1][0] = 2*ip;
    coeffs[1][1] = damp_cc[2] + 2*ip*damp_cc[0];
    coeffs[1][2] = damp_cc[3] + 2*ip*damp_cc[1];

    // stiffness
    PetscMalloc(sizeof(PetscReal)*3, &(coeffs[2]));
    //coeffs[2] = (PetscReal*)malloc(sizeof(PetscReal)*3);
    coeffs[2][0] = ip*ip;
    coeffs[2][1] = damp_cc[4] + ip*damp_cc[2] + ip*ip*damp_cc[0];
    coeffs[2][2] = damp_cc[5] + ip*damp_cc[3] + ip*ip*damp_cc[1];

    // DEBUG
    //PetscPrintf(comm, "DEBUG: Coeffs: \n");
    //PetscPrintf(comm, "%le, %le\n", coeffs[0][0], coeffs[0][1]);
    //PetscPrintf(comm, "%le, %le, %le\n", coeffs[1][0], coeffs[1][1], coeffs[1][2]);
    //PetscPrintf(comm, "%le, %le, %le\n", coeffs[2][0], coeffs[2][1], coeffs[2][2]);

    return coeffs;
}

void local_mat_mult(MPI_Comm comm, Mat *M, Mat *Dv, Mat *Dh, PetscReal **coeffs, Vec *qj, Vec *pj, Vec *result)
{
    // multiply qj with local damping mat and pj with local mass mat. put the result into 'result'.
    Vec result_tmp;

    //DEBUG
    //PetscReal norm;

    VecSet(*result, 0);
    MatCreateVecs(*M, &result_tmp, NULL);
    
    MatMult(*M, *qj, result_tmp);
    VecAXPY(*result, coeffs[1][0], result_tmp);

    // DEBUG
    //VecNorm(*result, NORM_2, &norm);
    //PetscPrintf(comm, "DEBUG: Norm1 = %le.\n", norm);

    MatMult(*Dv, *qj, result_tmp);
    VecAXPY(*result, coeffs[1][1], result_tmp);

    // DEBUG
    //VecNorm(*result, NORM_2, &norm);
    //PetscPrintf(comm, "DEBUG: Norm1 = %le.\n", norm);

    MatMult(*Dh, *qj, result_tmp);
    VecAXPY(*result, coeffs[1][2], result_tmp);

    // DEBUG
    //VecNorm(*result, NORM_2, &norm);
    //PetscPrintf(comm, "DEBUG: Norm1 = %le.\n", norm);

    MatMult(*M, *pj, result_tmp);
    VecAXPY(*result, 1, result_tmp);

    // DEBUG
    //VecNorm(*result, NORM_2, &norm);
    //PetscPrintf(comm, "DEBUG: Norm1 = %le.\n", norm);

    MatMult(*Dv, *pj, result_tmp);
    VecAXPY(*result, coeffs[0][0], result_tmp);

    // DEBUG
    //VecNorm(*result, NORM_2, &norm);
    //PetscPrintf(comm, "DEBUG: Norm1 = %le.\n", norm);

    MatMult(*Dh, *pj, result_tmp);
    VecAXPY(*result, coeffs[0][1], result_tmp);

    // DEBUG
    //VecNorm(*result, NORM_2, &norm);
    //PetscPrintf(comm, "DEBUG: Norm1 = %le.\n", norm);

    VecScale(*result, -1);

    // DEBUG
    //VecNorm(*result, NORM_2, &norm);
    //PetscPrintf(comm, "DEBUG: Norm1 = %le.\n", norm);

    VecDestroy(&result_tmp);
}


void soar(MPI_Comm comm, Mat *M, Mat *Dv, Mat *Dh, Mat *K, Vec *b, PetscInt n, PetscReal** coeffs, Vec *q, PetscInt *q_size)
{
    PetscPrintf(comm, "SOAR at interpolation point ...\n");

    Mat K0; // local stiffness matrix
    KSP ksp; PC pc; // solver and preconditioner context
    Vec *p; // *q; // Array of vectors for holding the Arnoldi basis
    Vec r, s; // Vectors to be used in SOAR loop
    Vec r_tmp_1; //, r_tmp_2;
    PetscReal r_norm; // for holding vector norm
    PetscScalar t_ij; // for holding dot product
    
    PetscInt i, j, k; // loop counters


    // generate local stiffness matrix
    PetscPrintf(comm, "Generating local stiffness matrix ...\n");
    MatDuplicate(*K, MAT_COPY_VALUES, &K0); // copy K to K0
    MatAXPY(K0, coeffs[2][0], *M, DIFFERENT_NONZERO_PATTERN);
    MatAXPY(K0, coeffs[2][1], *Dv, DIFFERENT_NONZERO_PATTERN);
    MatAXPY(K0, coeffs[2][2], *Dh, DIFFERENT_NONZERO_PATTERN);


    // allocate memory for vector arrays
    PetscMalloc(sizeof(Vec)*n, &p);
    //p = (Vec*)malloc(sizeof(Vec)*n);
    //q = (Vec*)malloc(sizeof(Vec)*n);

    KSPCreate(comm, &ksp);
    KSPSetOperators(ksp, K0, K0); // set operator as local stiffness matrix
    KSPSetType(ksp, KSPPREONLY); // pre-conditioner only (we want a direct solution)

    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU); // set preconditioner type to LU factorisation

    PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS); // set MUMPS as the LU solver

    KSPSetFromOptions(ksp);

    PetscPrintf(comm, "Solving for inital vector ...\n");
    MatCreateVecs(K0, &(q[0]), NULL); // create a right vector to solve the solution
    KSPSolve(ksp, *b, q[0]);
    PetscPrintf(comm, "Done. Continuing ...\n");

    VecNorm(q[0], NORM_2, &r_norm);
    VecScale(q[0], 1/r_norm);

    MatCreateVecs(K0, &(p[0]), NULL);
    VecSet(p[0], 0);

    *q_size = 1; // reset the size of the output

    MatCreateVecs(K0, &r, NULL);
    MatCreateVecs(K0, &s, NULL);
    MatCreateVecs(K0, &r_tmp_1, NULL);
    //MatCreateVecs(K0, &r_tmp_2, NULL);

    for(j=0; j<n-1; j++)
    {
        // multiply qj with local damping mat and pj with local mass mat
        local_mat_mult(comm, M, Dv, Dh, coeffs, &(q[j]), &(p[j]), &r_tmp_1);
        
        // DEBUG
        //VecNorm(r_tmp_1, NORM_2, &r_norm);
        //PetscPrintf(comm, "DEBUG: Norm of r_tmp_1 = %le\n", r_norm);

        //MatMult(*D, q[j], r_tmp_1);
        //MatMultAdd(*M, p[j], r_tmp_1, r_tmp_2);
        //VecScale(r_tmp_2, -1);
        //KSPSolve(ksp, r_tmp_2, r);

        PetscPrintf(comm, "Solving within loop #%d...\n", j+1);
        KSPSolve(ksp, r_tmp_1, r);
        VecCopy(q[j], s);

        // DEBUG
        //VecNorm(r, NORM_2, &r_norm);
        //PetscPrintf(comm, "DEBUG: Norm of r before = %le\n", r_norm);

        for(i=0; i<j; i++)
        {
            VecDot(r, q[i], &t_ij);
            VecAXPY(r, -t_ij, q[i]);
            VecAXPY(s, -t_ij, p[i]);
        }
        
        VecNorm(r, NORM_2, &r_norm);

        // DEBUG
        //PetscPrintf(comm, "DEBUG: Norm of r after = %le\n", r_norm);

        if(r_norm <= TOLERANCE)
        {
           PetscPrintf(comm, "Breakdown or Deflation. Not Implemented!. Exiting at j = %d.\n", j);

           // free memory
           MatDestroy(&K0);
           VecDestroy(&r); VecDestroy(&s); VecDestroy(&r_tmp_1); //VecDestroy(&r_tmp_2);
           for(k=0; k<n; ++k)
           {
               VecDestroy(&(p[k]));
           }
           PetscFree(p);

           // return basis
           //return q;
           return;
        }

        MatCreateVecs(K0, &(q[j+1]), NULL);
        VecScale(r, 1/r_norm);

        // DEBUG
        //VecNorm(r, NORM_2, &r_norm);
        //PetscPrintf(comm, "DEBUG: Norm of r after scaling = %le\n", r_norm);

        VecCopy(r, q[j+1]);

        // DEBUG
        //VecNorm(q[j+1], NORM_2, &r_norm);
        //PetscPrintf(comm, "DEBUG: Norm of q[j+1] = %le\n", r_norm);

        MatCreateVecs(K0, &(p[j+1]), NULL);
        VecScale(s, 1/r_norm);
        VecCopy(s, p[j+1]);

        (*q_size)++; // increment size of output array
    }

    KSPDestroy(&ksp);

    // free memory
    MatDestroy(&K0);
    VecDestroy(&r); VecDestroy(&s); VecDestroy(&r_tmp_1); //VecDestroy(&r_tmp_2);
    for(k=0; k<n; ++k)
    {
        VecDestroy(&(p[k]));
    }
    PetscFree(p);

    PetscPrintf(comm, "Done. Returning %d Arnoldi basis.\n", n);
    // return basis
    //return q;
}

void orthogonalize_arnoldi(MPI_Comm comm, Vec *q_old, PetscInt *n_old, Vec *q_new, PetscInt *n_new)
{
    PetscScalar t; // dot product holder
    PetscReal norm; 
    PetscInt i, j;

    PetscInt indices[*n_new], count;

    count = 0;

    // Gram-Schmidt orthogonalisation
    for(i = 0; i < (*n_new); i++)
    {
        for(j = 0; j < (*n_old); j++)
        {
            VecDot(q_old[j], q_new[i], &t);
            VecAXPY(q_new[i], t, q_old[j]);
        }

        VecNorm(q_new[i], NORM_2, &norm);

        if(norm > TOLERANCE)
        {
            indices[count] = i;
            count++;
        }
        else
        {
            VecDestroy(&(q_new[i]));
        }
    }

    // Grow existing arnoldi basis array
    q_old = (Vec*)realloc(q_old, sizeof(Vec)*((*n_old)+count));
    for(i=0; i<count; i++)
    {
        VecCopy(q_new[indices[i]], q_old[(*n_old)+i]);
        VecDestroy(&(q_new[i]));
    }

    // free pointers
    PetscFree(q_new);
    q_new = NULL;

    // increment the size of arnoldi basis
    (*n_old) += count;
}

void multi_soar(MPI_Comm comm, Mat *M, Mat *Dv, Mat *Dh, Mat *K, Vec *b, PetscInt n_ip, PetscInt n_arn, PetscReal *omega, PetscInt n_omega, Fitter *fits, PetscInt n_fits)
//Vec* multi_soar(MPI_Comm comm, Mat *M, Mat *Dv, Mat *Dh, Mat *K, Vec *b, PetscInt n_ip, PetscInt n_arn, PetscReal *omega, PetscInt n_omega, Fitter *fits, PetscInt n_fits)
{
    PetscReal *ind_tmp, **coeffs;
    PetscInt i, j, *interp_ind;
    //Vec *Q_old, *Q_new;
    //PetscInt nq_old, nq_new;
    Vec *Q;
    PetscInt n_q, n_q_tot=0;
    char filename[100];

    // find n_ip evenly distributed indices in omega
    ind_tmp = linspace(0, n_omega-1, n_ip+1);
    PetscMalloc(sizeof(PetscInt)*n_ip, &interp_ind);
    for(i=0; i<n_ip; i++)
    {
        interp_ind[i] = (PetscInt)round((ind_tmp[i] + ind_tmp[i+1])/2);
    }

    for(i=0; i<n_ip; i++)
    {
        PetscPrintf(comm, "Interpolation point = %f Hz [%d/%d] ...\n", omega[interp_ind[i]]/2/M_PI, i+1, n_ip);
        coeffs = generate_local_coeffs(comm, fits, n_fits, omega[interp_ind[i]], interp_ind[i]);
        
        /*if(i==0)
        {
            Q_old = soar(comm, M, Dv, Dh, K, b, n_arn, coeffs, &nq_old);
        }
        else
        {
            Q_new = soar(comm, M, Dv, Dh, K, b, n_arn, coeffs, &nq_new);
            orthogonalize_arnoldi(comm, Q_old, &nq_old, Q_new, &nq_new);
        }*/

        Q = (Vec*)malloc(sizeof(Vec)*n_arn);
        soar(comm, M, Dv, Dh, K, b, n_arn, coeffs, Q, &n_q);

        for(j=0; j<n_q; j++)
        {
            sprintf(filename, "arnoldi/union_%d.dat", n_q_tot+j);
            write_vec_file(comm, filename, &(Q[j]));
            VecDestroy(&(Q[j]));
        }
        PetscFree(Q); //Q = NULL;

        n_q_tot += n_q;
        
    }

    //return Q_old;

}

