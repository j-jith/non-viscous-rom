#ifndef GLOBALS_H
#define GLOBALS_H

//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>

// petsc headers
#include <petscsys.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <petscviewer.h>
#include <petscmath.h>

// slepc header
#include <slepceps.h>

#define TOLERANCE 1e-16
#define NFREQ 801

typedef struct {
    const char mass_file[100];
    const char stiff_file[100];
    const char damp_file[100];
    const char load_file[100];
    const char fit_file[100];
    const char weights_file[100];
} Params;

typedef struct {
    PetscReal c1[3];
    PetscReal c2[3];
    PetscReal *weights;
} Fitter;

// File I/O functions - file_io.c
void read_mat_file(MPI_Comm comm, const char filename[], Mat *A);
void read_vec_file(MPI_Comm comm, const char filename[], Vec *b);
void write_vec_file(MPI_Comm comm, const char filename[], Vec *b);
PetscInt count_rows(const char filename[]);
Fitter* read_fitter(const char fit_file[], const char weight_file[], PetscInt *fit_len);

// SOAR functions - multi_soar.c
PetscReal** generate_local_coeffs(MPI_Comm comm, Fitter *fits, PetscInt n_fits, PetscReal ip, PetscInt n_ip);

void local_mat_mult(MPI_Comm comm, Mat *M, Mat *C1, Mat *C2, PetscReal **coeffs, Vec *qj, Vec *pj, Vec *result);

void soar(MPI_Comm comm, Mat *M, Mat *C1, Mat *C2, Mat *K, Vec *b, PetscInt n, PetscReal** coeffs, Vec *q, PetscInt *q_size);
//Vec* soar(MPI_Comm comm, Mat *M, Mat *Dv, Mat *Dh, Mat *K, Vec *b, PetscInt n, PetscReal** coeffs, PetscInt *q_size);

void orthogonalize_arnoldi(MPI_Comm comm, Vec *q_old, PetscInt *n_old, Vec *q_new, PetscInt *n_new);

void multi_soar(MPI_Comm comm, Mat *M, Mat *C1, Mat *C2, Mat *K, Vec *b, PetscInt n_ip, PetscInt n_arn, PetscReal *omega, PetscInt n_omega, Fitter *fits, PetscInt n_fits);
//Vec* multi_soar(MPI_Comm comm, Mat *M, Mat *Dv, Mat *Dh, Mat *K, Vec *b, PetscInt n_ip, PetscInt n_arn, PetscReal *omega, PetscInt n_omega, Fitter *fits, PetscInt n_fits);



// Construct full (block) matrices from smaller matrices - block_matrices.c
void get_csr(MPI_Comm comm, Mat *M,
        PetscInt **ai, PetscInt **aj, PetscScalar **av,
        PetscInt *total_rows);

void get_block_diag_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1, PetscBool flip_signs);

void get_block_mass_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1);

void get_block_stiffness_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1);

void get_block_damping_csr(MPI_Comm comm,
        PetscInt *ai, PetscInt *aj, PetscScalar *av, PetscInt nrows,
        PetscInt **ai1, PetscInt **aj1, PetscScalar **av1,
        PetscInt **ai2, PetscInt **aj2, PetscScalar **av2);

void create_block_mass(MPI_Comm comm, Mat *M, Mat *M1);
void create_block_stiffness(MPI_Comm comm, Mat *M, Mat *M1);
void create_block_damping(MPI_Comm comm, Mat *M, Mat *M1, Mat *M2);
void create_block_load(MPI_Comm comm, Vec *f, Vec *f1);

// POD functions for orthogonalisation
void get_covariance(MPI_Comm comm, Vec *Q, PetscInt n, Mat *R);
void get_pod_eigenvectors(MPI_Comm comm, Mat *A, PetscScalar tol,
        Vec **xr, PetscInt *rank);
void pod_orthogonalise(MPI_Comm comm, Vec *Q, PetscInt n_q, PetscScalar tol,
        Vec **Q1, PetscInt *rank);
void check_orthogonality(MPI_Comm comm, Vec *Q, PetscInt n_q);

// Reduce and solve - reduce.c
void direct_solve_dense(MPI_Comm comm, Mat *A, Vec *b, Vec *u);
void project_matrix(MPI_Comm comm, Mat *M, Vec *Q, PetscInt n_q, Mat *A);
void project_vector(MPI_Comm comm, Vec *u, Vec *Q, PetscInt n_q, Vec *u_new);

void direct_sweep_approx(MPI_Comm comm, Mat *M, Mat *C1, Mat *C2, Mat *K, Vec *b,
        PetscScalar omega_i, PetscScalar omega_f, PetscInt n_omega,
        Fitter *fits, PetscInt n_fits, Vec **u);

void direct_sweep(MPI_Comm comm, Mat *M, Mat *C1, Mat *C2, Mat *K, Vec *b,
        PetscScalar omega_i, PetscScalar omega_f, PetscInt n_omega,
        PetscScalar mu, Vec **u);

void recover_vector(MPI_Comm comm, Vec *u, Vec *Q, PetscInt n_q, Vec *u_new);
void recover_vectors(MPI_Comm comm, Vec *u, PetscInt n_u, Vec *Q, PetscInt n_q,
        Vec **u_new);

// Miscellaneous - misc.c
PetscReal* linspace(PetscReal start, PetscReal stop, PetscInt len);

#endif
