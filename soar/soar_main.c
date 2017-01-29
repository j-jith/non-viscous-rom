#include "globals.h"

int main(int argc, char **args)
{
    PetscInt pod_tolerance = 0;
    //PetscScalar mu = 4129.28;
    
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
    Vec *Q, *Q1=NULL;
    //Vec *Q_tmp;
    PetscInt n_q=0, n_q_tot=0, pod_rank=0;

    Mat M0, K0, C0; // Small matrices
    Mat M, K, C1, C2; // Big matrices
    Vec b0; // Small vectors
    Vec b, *u_full=NULL; // Big vectors

    // Reduced matrices
    Mat Mr, Kr, C1r, C2r;
    Vec br, *ur=NULL;

    PetscInt i, j;
    //PetscErrorCode ierr;

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
    //PetscReal jk;

    if (n_ip == 1)
    {
        coeffs = generate_local_coeffs(MPI_COMM_WORLD, fit_list, n_fits,
                omega[ind_ip[0]], ind_ip[0]);

        PetscMalloc(sizeof(Vec)*n_arn, &Q);
        soar(MPI_COMM_WORLD, &M, &C1, &C2, &K, &b, n_arn, coeffs, Q, &n_q_tot);
        Q1 = Q;
        pod_rank = n_q_tot;
    }
    else
    {
        PetscMalloc(sizeof(Vec)*n_ip*n_arn, &Q);

        // Multi-point SOAR loop
        for(j=0; j<n_ip; j++)
        {
            // Generate fit coeffs at interpolation point
            coeffs = generate_local_coeffs(MPI_COMM_WORLD, fit_list, n_fits,
                    omega[ind_ip[j]], ind_ip[j]);

            soar(MPI_COMM_WORLD, &M, &C1, &C2, &K, &b, n_arn, coeffs,
                    &(Q[n_q_tot]), &n_q);

            /* // Write basis vectors to disk
               for(i=0; i<n_q; i++)
               {
               sprintf(q_file, "output/arnoldi/basis_%d_%d.dat", ind_ip[j], i);
               write_vec_file(MPI_COMM_WORLD, q_file, &(Q[i+n_q_tot]));
            //VecDestroy(&(Q[i]));
            }
            */

            /*
            for(i=n_q; i<n_arn; i++)
                VecDestroy(&(Q[i+n_q_tot]));
            */

            /*
            VecNorm(Q[n_q_tot+n_q-1], NORM_1, &jk);
            PetscPrintf(PETSC_COMM_WORLD, "*** norm(%d) = %e ***\n", n_q_tot+n_q-1, jk);
            */

            n_q_tot += n_q;

            // In-loop clean-up
            //PetscFree(Q);
            PetscFree(coeffs);
        }


        // Get new basis vectors using POD orthogonalisation
        pod_orthogonalise(PETSC_COMM_WORLD, Q, n_q_tot, pod_tolerance, &Q1, &pod_rank);
        // Free the old basis vector memory
        for(i=0; i<n_q_tot; i++)
        {
            VecDestroy(&(Q[i]));
        }
        PetscFree(Q);

    }


    // DEBUG: check if new basis vectors are orthogonal
    //check_orthogonality(PETSC_COMM_WORLD, Q1, pod_rank);

    // DEBUG: check vector projection/recovery
    //check_projection(PETSC_COMM_WORLD, b, Q1, pod_rank);


    // Generate reduced matrices
    PetscPrintf(PETSC_COMM_WORLD, "Generating reduced matrices...\n");
    project_matrix(PETSC_COMM_WORLD, &M, Q1, pod_rank, &Mr);
    project_matrix(PETSC_COMM_WORLD, &K, Q1, pod_rank, &Kr);
    project_matrix(PETSC_COMM_WORLD, &C1, Q1, pod_rank, &C1r);
    project_matrix(PETSC_COMM_WORLD, &C2, Q1, pod_rank, &C2r);
    project_vector(PETSC_COMM_WORLD, &b, Q1, pod_rank, &br);
    PetscPrintf(PETSC_COMM_WORLD, "Done\n");

    // Free big matrices
    MatDestroy(&M);
    MatDestroy(&K);
    MatDestroy(&C1);
    MatDestroy(&C2);
    VecDestroy(&b);

    /*
    // DEBUG: Write reduced matrices to disk
    write_mat_file(PETSC_COMM_WORLD, "redmats/mass.dat", &Mr);
    write_mat_file(PETSC_COMM_WORLD, "redmats/damp1.dat", &C1r);
    write_mat_file(PETSC_COMM_WORLD, "redmats/damp2.dat", &C2r);
    write_mat_file(PETSC_COMM_WORLD, "redmats/stiff.dat", &Kr);
    write_vec_file(PETSC_COMM_WORLD, "redmats/load.dat", &br);
    */

    // Solve reduced problem
    PetscPrintf(PETSC_COMM_WORLD, "Frequency sweep of reduced system...\n");

    direct_sweep(PETSC_COMM_WORLD, &Mr, &C1r, &C2r, &Kr, &br,
        omega_i, omega_f, omega_len, mu, &ur);
    
    //direct_sweep_piecewise(PETSC_COMM_WORLD, &Mr, &C1r, &C2r, &Kr, &br,
    //    omega_i, omega_f, omega_len, fit_list, n_fits, &ur);
    
    PetscPrintf(PETSC_COMM_WORLD, "Done\n");

    // Free reduced matrices
    MatDestroy(&Mr);
    MatDestroy(&Kr);
    MatDestroy(&C1r);
    MatDestroy(&C2r);
    VecDestroy(&br);

    // Recover full solution
    PetscPrintf(PETSC_COMM_WORLD, "Recovering full solution...\n");
    recover_vectors(PETSC_COMM_WORLD, ur, omega_len, Q1, pod_rank, &u_full);
    PetscPrintf(PETSC_COMM_WORLD, "Done\n");

    // Free the new basis vector memory
    for(i=0; i<pod_rank; i++)
    {
        VecDestroy(&(Q1[i]));
    }
    PetscFree(Q1);

    // Write solution to disk
    for(i=0; i<omega_len; i++)
    {
        sprintf(q_file, "output/red/solution/%d.dat", i);
        write_vec_file(MPI_COMM_WORLD, q_file, &(u_full[i]));
        VecDestroy(&(u_full[i]));

        VecDestroy(&(ur[i]));
    }
    PetscFree(u_full);
    PetscFree(ur);


    SlepcFinalize();
    //PetscFinalize();

    return 0;
}
