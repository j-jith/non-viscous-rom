#include "globals.h"

int main(int argc, char **args)
{
    char mass_file[] = "../matrices/mass.dat";
    char stiff_file[] = "../matrices/stiffness.dat";
    char damp_file[] = "../matrices/damping.dat";
    char load_file[] = "../matrices/force.dat";
    char q_file[100];

    char fit_file[] = "../fitting/fit_list.txt";
    char weights_file[] = "../fitting/weights.txt";

    PetscReal omega_i, omega_f;
    PetscInt omega_len, n_fits=0;
    //PetscReal ip;
    PetscInt n_arn, n_ip;
    PetscReal *omega;
    PetscInt *ind_ip;
    PetscReal *ind_ip_tmp;

    Fitter *fit_list;
    PetscReal **coeffs;
    Vec *Q, *Q_tmp, *Q1=NULL;
    PetscInt n_q=0, n_q_tot=0, pod_rank=0;

    Mat M0, K0, C0; // Small matrices
    Mat M, K, C1, C2; // Big matrices
    Vec b0; // Small vectors
    Vec b; // Big vectors

    PetscInt i, j;
    PetscErrorCode ierr;

    //PetscInitialize(&argc, &args, NULL, NULL);
    SlepcInitialize(&argc, &args, NULL, NULL);

    if(argc > 5)
    {
        omega_i = (PetscReal)atof(args[1])*2*M_PI;
        omega_f = (PetscReal)atof(args[2])*2*M_PI;
        omega_len = (PetscInt)atoi(args[3]);
        n_ip = (PetscInt)atoi(args[4]);
        n_arn = (PetscInt)atoi(args[5]);
    }
    else
    {
        //ind_ip = (PetscInt)round(omega_len/2);
        PetscPrintf(MPI_COMM_WORLD, "Usage: ./soar <initial freq.> <final freq.> <no. of freqs.> <no. of interpolation points> <no. of basis vectors per interpolation point>\n");
        return 0;
    }

    omega = linspace(omega_i, omega_f, omega_len);
    ind_ip_tmp = linspace(0, omega_len-1, n_ip+1);
    PetscMalloc1(n_ip, &ind_ip);
    for(i=0; i<n_ip; i++)
    {
        ind_ip[i] = (PetscInt)round((ind_ip_tmp[i]+ind_ip_tmp[i+1])/2);
    }

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
    fit_list = read_fitter(fit_file, weights_file, &n_fits);

    Q = NULL; n_q = 0; n_q_tot = 0;
    // Multi-point SOAR loop
    for(j=0; j<n_ip; j++)
    {
        // Generate fit coeffs at interpolation point
        coeffs = generate_local_coeffs(MPI_COMM_WORLD, fit_list, n_fits, omega[ind_ip[j]], ind_ip[j]);

        //PetscMalloc(sizeof(Vec)*n_arn, &Q);
        //soar(MPI_COMM_WORLD, &M, &C1, &C2, &K, &b, n_arn, coeffs, Q, &n_q);
        //for(i=0; i<n_q; i++)
        //{
        //    sprintf(q_file, "arnoldi/basis_%d_%d.dat", ind_ip[j], i);
        //    write_vec_file(MPI_COMM_WORLD, q_file, &(Q[i]));
        //    VecDestroy(&(Q[i]));
        //}
        //for(i=n_q; i<n_arn; i++)
        //    VecDestroy(&(Q[i]));

        //ierr = PetscRealloc(sizeof(Vec)*n_arn, &Q);
        Q_tmp = realloc(Q, sizeof(Vec)*(n_q_tot+n_arn));
        if(Q_tmp == NULL)
        {
            PetscPrintf(MPI_COMM_WORLD, "Error during REALLOC\n");
            return -1;
        }
        else
            Q = Q_tmp;

        soar(MPI_COMM_WORLD, &M, &C1, &C2, &K, &b, n_arn, coeffs, &(Q[n_q]), &n_q);

        for(i=0; i<n_q; i++)
        {
            sprintf(q_file, "arnoldi/basis_%d_%d.dat", ind_ip[j], i);
            write_vec_file(MPI_COMM_WORLD, q_file, &(Q[i+n_q_tot]));
            //VecDestroy(&(Q[i]));
        }
        for(i=n_q; i<n_arn; i++)
            VecDestroy(&(Q[i+n_q_tot]));

        n_q_tot += n_q;

        // In-loop clean-up
        //PetscFree(Q);
        PetscFree(coeffs);
    }

    // Get new basis vectors using POD orthogonalisation
    pod_orthogonalise(PETSC_COMM_WORLD, Q, n_q_tot, 1e-12, Q1, &pod_rank);

    // Free the old basis vector memory
    for(i=0; i<n_q_tot; i++)
    {
        VecDestroy(&(Q[i]));
    }
    PetscFree(Q);
    PetscFree(Q_tmp);

    // Free work space
    MatDestroy(&M);
    MatDestroy(&K);
    MatDestroy(&C1);
    MatDestroy(&C2);
    VecDestroy(&b);


    SlepcFinalize();
    //PetscFinalize();

    return ierr;
}
