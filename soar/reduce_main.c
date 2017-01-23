#include "globals.h"
#include <petscmath.h>

int main(int argc, char **args)
{
    char basis_dir[] = "arnoldi";
    unsigned int n_arn = 10;
    unsigned int len_ip = 10, *ind_ip;

    char mass_file[] = "../mass_matrix.dat";
    char stiff_file[] = "../stiffness_matrix.dat";
    char damp_v_file[] = "../damping_v_matrix.dat";
    char damp_h_file[] = "../damping_h_matrix.dat";
    char load_file[] = "../load_vector.dat";

    char mass_out[] = "mass_red.txt";
    char stiff_out[] = "stiffness_red.txt";
    char damp_v_out[] = "damping_v_red.txt";
    char damp_h_out[] = "damping_h_red.txt";
    char load_out[] = "load_red.txt";

    char ortho_dir[] = "arnoldi/ortho";
    char ortho_file[100];

    //char sol_dir[] = "reduced";
    //char sol_file[100];

/*    // Fluid properties
    // Density of fluid material
    PetscReal rho_0 = 688.34;
    // Speed of sound in fluid
    PetscReal c_0   = 376.05;
    // Isobaric heat capacity of fluid
    PetscReal c_p    = 2739.8;
    // Ratio of heat capacities of fluid
    PetscReal gamma = 2.9408;
    // Thermal conductivity of fluid
    PetscReal kappa = 0.074537;
    // Viscosity of fluid
    PetscReal mu    = 5.5483e-5;
    // Damping coefficients
    PetscReal alpha_v = -1/sqrt(2)*sqrt(mu/rho_0);
    PetscReal alpha_h = 1/sqrt(2)*(gamma-1)/c_0/c_0*sqrt(kappa/rho_0/c_p);

    PetscReal omega_i = 2*M_PI*150;
    PetscReal omega_f = 2*M_PI*950;
    unsigned int omega_len = 801;
    PetscReal *omega;

    Mat Mr, Dvr, Dhr, Kr, Ar;
    Vec br, ur; */

    Vec *Q;
    unsigned int q_len;

    unsigned int i;

    Q = (Vec*)malloc(sizeof(Vec)*len_ip*n_arn);

    ind_ip = (unsigned int*)malloc(sizeof(unsigned int)*len_ip);
    for(i=0; i<len_ip; i++)
    {
        ind_ip[i] = 40 + i*80;
    }

    PetscInitialize(&argc, &args, NULL, NULL);

    orthogonalize_arnoldi_disk(MPI_COMM_WORLD, basis_dir, ind_ip, len_ip, n_arn, Q, &q_len);

    //write ortho basis to file
    PetscPrintf(PETSC_COMM_WORLD, "Writing ortho Arnoldi basis to file...\n");
    for(i=0; i<q_len; i++)
    {
        sprintf(ortho_file, "%s/ortho_%d.dat", ortho_dir, i);
        write_vec_file(PETSC_COMM_WORLD, ortho_file, &(Q[i])); 
        strcpy(ortho_file, "");
    }

    PetscPrintf(PETSC_COMM_WORLD, "Generating reduced mass matrix...\n");
    generate_reduced_matrix(MPI_COMM_WORLD, mass_file, Q, q_len, mass_out);

    PetscPrintf(PETSC_COMM_WORLD, "Generating reduced v-damping matrix...\n");
    generate_reduced_matrix(MPI_COMM_WORLD, damp_v_file, Q, q_len, damp_v_out);

    PetscPrintf(PETSC_COMM_WORLD, "Generating reduced h-damping matrix...\n");
    generate_reduced_matrix(MPI_COMM_WORLD, damp_h_file, Q, q_len, damp_h_out);
    
    PetscPrintf(PETSC_COMM_WORLD, "Generating reduced stiffness matrix...\n");
    generate_reduced_matrix(MPI_COMM_WORLD, stiff_file, Q, q_len, stiff_out);

    PetscPrintf(PETSC_COMM_WORLD, "Generating reduced load vector...\n");
    generate_reduced_vector(MPI_COMM_WORLD, load_file, Q, q_len, load_out);

    /*MatCreate(MPI_COMM_WORLD, &Mr);
    MatCreate(MPI_COMM_WORLD, &Dvr);
    MatCreate(MPI_COMM_WORLD, &Dhr);
    MatCreate(MPI_COMM_WORLD, &Kr);
    MatSetSizes(Mr, PETSC_DECIDE, PETSC_DECIDE, q_len, q_len);
    MatSetSizes(Dvr, PETSC_DECIDE, PETSC_DECIDE, q_len, q_len);
    MatSetSizes(Dhr, PETSC_DECIDE, PETSC_DECIDE, q_len, q_len);
    MatSetSizes(Kr, PETSC_DECIDE, PETSC_DECIDE, q_len, q_len);
    MatSetType(Mr, MATDENSE);
    MatSetType(Dvr, MATDENSE);
    MatSetType(Dhr, MATDENSE);
    MatSetType(Kr, MATDENSE);
    MatSetFromOptions(Mr);
    MatSetFromOptions(Dvr);
    MatSetFromOptions(Dhr);
    MatSetFromOptions(Kr);
    MatSetUp(Mr);
    MatSetUp(Dvr);
    MatSetUp(Dhr);
    MatSetUp(Kr);

    PetscPrintf(PETSC_COMM_WORLD, "Generating reduced mass matrix...\n");
    generate_reduced_matrix(MPI_COMM_WORLD, mass_file, Q, q_len, &Mr);

    PetscPrintf(PETSC_COMM_WORLD, "Generating reduced v-damping matrix...\n");
    generate_reduced_matrix(MPI_COMM_WORLD, damp_v_file, Q, q_len, &Dvr);

    PetscPrintf(PETSC_COMM_WORLD, "Generating reduced h-damping matrix...\n");
    generate_reduced_matrix(MPI_COMM_WORLD, damp_h_file, Q, q_len, &Dhr);
    
    PetscPrintf(PETSC_COMM_WORLD, "Generating reduced stiffness matrix...\n");
    generate_reduced_matrix(MPI_COMM_WORLD, stiff_file, Q, q_len, &Kr);

    PetscPrintf(PETSC_COMM_WORLD, "Generating reduced load vector...\n");
    generate_reduced_vector(MPI_COMM_WORLD, load_file, Q, q_len, &br);

    // create frequency array
    omega = linspace(omega_i, omega_f, omega_len);

    MatDuplicate(Kr, MAT_DO_NOT_COPY_VALUES, &Ar);
    MatCreateVecs(Kr, &ur, NULL);

    for(i=0; i<omega_len; i++)
    {
        PetscPrintf(PETSC_COMM_WORLD, "Frequency = %f Hz [%d/%d]", omega[i]/2/M_PI, i+1, omega_len);

        PetscPrintf(PETSC_COMM_WORLD, "Adding up LHS matrices ...\n");
        MatCopy(Kr, Ar, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(Ar, omega[i]*omega[i], Mr, DIFFERENT_NONZERO_PATTERN);
        //MatAXPY(Ar, alpha_v/PetscSqrtReal(omega[i]), Dvr, DIFFERENT_NONZERO_PATTERN);
        //MatAXPY(Ar, alpha_h*omega[i]*PetscSqrtReal(omega[i]), Dhr, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(Ar, alpha_v/sqrt(omega[i]), Dvr, DIFFERENT_NONZERO_PATTERN);
        MatAXPY(Ar, alpha_h*omega[i]*sqrt(omega[i]), Dhr, DIFFERENT_NONZERO_PATTERN);

        // solve
        direct_solve_dense(MPI_COMM_WORLD, &Ar, &br, &ur); 
        
        // write to file
        sprintf(sol_file, "%s/red_sol_%d.dat", sol_dir, i);
        write_vec_file(MPI_COMM_WORLD, sol_file, &ur);
    }

    // free memory
    for(i=0; i<q_len; i++)
    {
        VecDestroy(&(Q[i]));
    }
    free(Q);
    MatDestroy(&Ar);
    MatDestroy(&Mr); MatDestroy(&Dvr);
    MatDestroy(&Kr); MatDestroy(&Dhr);
    VecDestroy(&br);
    VecDestroy(&ur);*/

    PetscFinalize();
    return 0;
}
