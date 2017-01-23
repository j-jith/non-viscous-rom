#ifndef GLOBALS_H
#define GLOBALS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// petsc headers
#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <petscviewer.h>
//#include <petscmath.h>

#define TOLERANCE 1e-16
#define NFREQ 801

typedef struct {
    PetscReal c1[3];
    PetscReal c2[3];
    PetscReal *weights;
} Fitter;

// File I/O functions - file_io.c
void read_mat_file(MPI_Comm comm, const char filename[], Mat *A);
void read_vec_file(MPI_Comm comm, const char filename[], Vec *b);
void write_vec_file(MPI_Comm comm, const char filename[], Vec *b);
unsigned int count_rows(const char filename[]);
Fitter* read_fitter(const char fit_file[], const char weight_file[], unsigned int *fit_len);

// SOAR functions - multi_soar.c
PetscReal** generate_local_coeffs(MPI_Comm comm, Fitter *fits, unsigned int n_fits, PetscReal ip, unsigned int n_ip);
void local_mat_mult(MPI_Comm comm, Mat *M, Mat *Dv, Mat *Dh, PetscReal **coeffs, Vec *qj, Vec *pj, Vec *result);
void soar(MPI_Comm comm, Mat *M, Mat *Dv, Mat *Dh, Mat *K, Vec *b, unsigned int n, PetscReal** coeffs, Vec *q, unsigned int *q_size);
//Vec* soar(MPI_Comm comm, Mat *M, Mat *Dv, Mat *Dh, Mat *K, Vec *b, unsigned int n, PetscReal** coeffs, unsigned int *q_size);
void orthogonalize_arnoldi(MPI_Comm comm, Vec *q_old, unsigned int *n_old, Vec *q_new, unsigned int *n_new);
void multi_soar(MPI_Comm comm, Mat *M, Mat *Dv, Mat *Dh, Mat *K, Vec *b, unsigned int n_ip, unsigned int n_arn, PetscReal *omega, unsigned int n_omega, Fitter *fits, unsigned int n_fits);
//Vec* multi_soar(MPI_Comm comm, Mat *M, Mat *Dv, Mat *Dh, Mat *K, Vec *b, unsigned int n_ip, unsigned int n_arn, PetscReal *omega, unsigned int n_omega, Fitter *fits, unsigned int n_fits);

// Reduce and solve - reduce.c
void read_arnoldi_basis(MPI_Comm comm, const char dirname[], unsigned int *ind_ip, unsigned int len_ip, unsigned int n_arn, Vec *Q);
//void generate_reduced_matrix(MPI_Comm comm, const char filename[], Vec *Q, unsigned int len_q, Mat *A);
//void generate_reduced_vector(MPI_Comm comm, const char filename[], Vec *Q, unsigned int len_q, Vec *b);
void generate_reduced_matrix(MPI_Comm comm, const char filename[], Vec *Q, unsigned int len_q, const char outfile[]);
void generate_reduced_vector(MPI_Comm comm, const char filename[], Vec *Q, unsigned int len_q, const char outfile[]);
void orthogonalize_arnoldi_disk(MPI_Comm comm, const char dirname[], unsigned int *ind_ip, unsigned int len_ip, unsigned int n_arn, Vec *Q, unsigned int *q_len);
void direct_solve_dense(MPI_Comm comm, Mat *A, Vec *b, Vec *u);

// Miscellaneous - misc.c
PetscReal* linspace(PetscReal start, PetscReal stop, unsigned int len);

#endif
