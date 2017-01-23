#ifndef GLOBALS_H
#define GLOBALS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// petsc headers
#include <petscsys.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <petscviewer.h>
//#include <petscmath.h>

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

void get_covariance(MPI_Comm comm, Vec *Q, PetscInt n, Mat *R);

// Reduce and solve - reduce.c
void read_arnoldi_basis(MPI_Comm comm, const char dirname[], PetscInt *ind_ip, PetscInt len_ip, PetscInt n_arn, Vec *Q);
//void generate_reduced_matrix(MPI_Comm comm, const char filename[], Vec *Q, PetscInt len_q, Mat *A);
//void generate_reduced_vector(MPI_Comm comm, const char filename[], Vec *Q, PetscInt len_q, Vec *b);
void generate_reduced_matrix(MPI_Comm comm, const char filename[], Vec *Q, PetscInt len_q, const char outfile[]);
void generate_reduced_vector(MPI_Comm comm, const char filename[], Vec *Q, PetscInt len_q, const char outfile[]);
void orthogonalize_arnoldi_disk(MPI_Comm comm, const char dirname[], PetscInt *ind_ip, PetscInt len_ip, PetscInt n_arn, Vec *Q, PetscInt *q_len);
void direct_solve_dense(MPI_Comm comm, Mat *A, Vec *b, Vec *u);

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


// Miscellaneous - misc.c
PetscReal* linspace(PetscReal start, PetscReal stop, PetscInt len);

#endif
