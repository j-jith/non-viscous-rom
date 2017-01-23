#include "globals.h"

int main(int argc, char **args)
{
    char mass_file[] = "../mass_matrix.dat";
    char stiff_file[] = "../stiffness_matrix.dat";
    char damp_v_file[] = "../damping_v_matrix.dat";
    char damp_h_file[] = "../damping_h_matrix.dat";
    char load_file[] = "../load_vector.dat";
    char q_file[100];

    char fit_file[] = "../fit_list.txt";
    char weights_file[] = "../weights.txt";

    PetscReal omega_i = 2*M_PI*150;
    PetscReal omega_f = 2*M_PI*950;
    unsigned int omega_len = 801;
    unsigned int ind_ip=0, n_fits=0;
    //PetscReal ip;
    unsigned int n_arn = 20;
    PetscReal *omega;

    Fitter *fit_list;
    PetscReal **coeffs;
    Vec *Q;
    unsigned int n_q=0;

    Mat *M, *K, *Dv, *Dh; // Matrices
    Vec *b; // Vectors

    unsigned int i;

    PetscInitialize(&argc, &args, NULL, NULL);

    omega = linspace(omega_i, omega_f, omega_len);
    if(argc > 2)
    {
        ind_ip = (unsigned int)atoi(args[1]);
        n_arn = (unsigned int)atoi(args[2]);
    }
    else
    {
        //ind_ip = (unsigned int)round(omega_len/2);
        PetscPrintf(MPI_COMM_WORLD, "ERROR: Please provide ind_ip and n_arn as arguments!\n");
        return 0;
    }
    
    M = (Mat*)malloc(sizeof(Mat));
    Dv = (Mat*)malloc(sizeof(Mat));
    Dh = (Mat*)malloc(sizeof(Mat));
    K = (Mat*)malloc(sizeof(Mat));
    b = (Vec*)malloc(sizeof(Vec));

    read_mat_file(MPI_COMM_WORLD, mass_file, M);
    read_mat_file(MPI_COMM_WORLD, stiff_file, K);
    read_mat_file(MPI_COMM_WORLD, damp_v_file, Dv);
    read_mat_file(MPI_COMM_WORLD, damp_h_file, Dh);
    read_vec_file(MPI_COMM_WORLD, load_file, b);

    fit_list = read_fitter(fit_file, weights_file, &n_fits);
    coeffs = generate_local_coeffs(MPI_COMM_WORLD, fit_list, n_fits, omega[ind_ip], ind_ip);
    
    Q = (Vec*)malloc(sizeof(Vec)*n_arn);
    soar(MPI_COMM_WORLD, M, Dv, Dh, K, b, n_arn, coeffs, Q, &n_q);

    for(i=0; i<n_q; i++)
    {
        sprintf(q_file, "arnoldi/basis_%d_%d.dat", ind_ip, i);
        write_vec_file(MPI_COMM_WORLD, q_file, &(Q[i]));
        VecDestroy(&(Q[i]));
    }
    free(Q); //Q = NULL;

    // Free work space
    MatDestroy(M); free(M);
    MatDestroy(K); free(K);
    MatDestroy(Dv); free(Dv);
    MatDestroy(Dh); free(Dh);
    VecDestroy(b); free(b);

    /*for(i=0; i<n_arn; i++)
    {
        VecDestroy(&(Q[0]));
    }
    free(Q);*/

    return 0;
}
